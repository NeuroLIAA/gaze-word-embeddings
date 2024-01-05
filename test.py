import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from gensim.models import KeyedVectors
import scripts.utils as utils


def test(model_path, wa_file, sa_file, wf_file, min_freq, num_samples, sim_threshold, gt_threshold, gt_embeddings_file,
         stimuli_path, save_path, sort_sim_by, error_bars):
    models = [dir_ for dir_ in sorted(model_path.iterdir()) if dir_.is_dir()]
    if len(models) == 0:
        raise ValueError(f'There are no models in {model_path}')
    if not (sa_file.exists() or wa_file.exists()):
        raise ValueError(f'Evaluation files missing: {wa_file} and {sa_file} do not exist')
    if not stimuli_path.exists():
        raise ValueError(f'Stimuli files missing: {stimuli_path} does not exist')
    subjs_associations = pd.read_csv(sa_file, index_col=0)
    words_associations, words_frequency = pd.read_csv(wa_file), pd.read_csv(wf_file)
    words_associations = words_associations[words_associations['cue'].isin(
        utils.subsample(words_associations['cue'], num_samples, seed=42))]
    gt_embeddings = KeyedVectors.load_word2vec_format(str(gt_embeddings_file))
    words_in_stimuli = utils.get_words_in_corpus(stimuli_path)
    models_results = {'similarity_to_subjs': {}, 'similarity_to_answers': {}, 'word_pairs': {}, 'gt_similarities': {}}
    for model_dir in models:
        model_wv = KeyedVectors.load_word2vec_format(str(model_dir / f'{model_dir.name}.vec'))
        test_model(model_wv, model_dir.name, words_associations, words_frequency, subjs_associations, gt_embeddings,
                   words_in_stimuli, models_results)

    model_basename = model_path.name
    save_path = save_path / model_basename
    save_path.mkdir(exist_ok=True, parents=True)
    plot_similarity(model_basename, models_results['similarity_to_subjs'], sim_threshold,
                    save_path, sort_sim_by, error_bars)
    plot_freq_to_sim(model_basename, models_results['similarity_to_answers'], save_path, min_appearances=min_freq)
    plot_distance_to_gt(model_basename, models_results['gt_similarities'], sim_threshold, gt_threshold,
                        save_path, error_bars)
    print_words_pairs_correlations(models_results['word_pairs'])


def test_model(model_wv, model_name, words_associations, words_frequency, subjs_associations, gt_embeddings,
               words_in_stimuli, models_results):
    answers_sim = utils.similarities(model_wv, words_associations['cue'], words_associations['answer'])
    wa_model_sim = words_associations.copy()
    wa_model_sim['similarity'] = answers_sim
    words = words_associations['cue'].drop_duplicates()
    words = words_frequency[words_frequency['word'].isin(words)]
    subjs_cues = subjs_associations.index
    sa_subj_sim = subjs_associations.copy().apply(lambda answers: utils.similarities(model_wv, subjs_cues, answers))
    models_results['similarity_to_subjs'][model_name] = sa_subj_sim
    models_results['similarity_to_answers'][model_name] = wa_model_sim
    models_results['word_pairs'][model_name] = evaluate_word_pairs(model_wv, wa_model_sim)
    models_results['gt_similarities'][model_name] = gt_similarities(model_wv, words, words_in_stimuli, gt_embeddings)


def gt_similarities(words_vectors, cues, words_in_stimuli, gt_embeddings, n=100):
    in_stimuli, off_stimuli = cues[cues['word'].isin(words_in_stimuli)], cues[~cues['word'].isin(words_in_stimuli)]
    in_stimuli = utils.subsample(in_stimuli, n, seed=42)
    matched_words = []
    for log_cnt in in_stimuli['log_cnt']:
        matched_word = off_stimuli.iloc[(off_stimuli['log_cnt'] - log_cnt).abs().argsort()[:1]]
        matched_words.append(matched_word['word'].sample(1, random_state=42).values[0])
        off_stimuli = off_stimuli[off_stimuli['word'] != matched_words[-1]]
    off_stimuli = cues[cues['word'].isin(matched_words)]
    in_stimuli, off_stimuli = utils.build_all_pairs(in_stimuli['word']), utils.build_all_pairs(off_stimuli['word'])
    in_stimuli['in_stimuli'], off_stimuli['in_stimuli'] = True, False

    gt_similarities = pd.concat([in_stimuli, off_stimuli])
    gt_similarities = gt_similarities[gt_similarities['cue'] != gt_similarities['answer']]
    gt_similarities['sim'] = utils.similarities(words_vectors, gt_similarities['cue'], gt_similarities['answer'])
    gt_similarities['sim_gt'] = utils.similarities(gt_embeddings, gt_similarities['cue'], gt_similarities['answer'])

    return gt_similarities


def evaluate_word_pairs(words_vectors, freq_similarity_pairs):
    temp_file = Path('word_pairs.csv')
    word_pairs = freq_similarity_pairs.drop(columns=['similarity', 'n'])
    word_pairs.to_csv(temp_file, index=False, header=False)
    pearson, spearman, oov_ratio = words_vectors.evaluate_word_pairs(temp_file, delimiter=',')
    temp_file.unlink()
    return pearson, spearman, [oov_ratio]


def print_words_pairs_correlations(models_results):
    print('\n---Correlation between similarity and frequency of response for cue-answer pairs---')
    measures = ['Pearson', 'Spearman rank-order', 'Out of vocabulary ratio']
    for i, measure in enumerate(measures):
        print(f'{measure}')
        for model in models_results:
            model_correlations = models_results[model][i]
            print(f'{model}: {model_correlations[0]:.4f}', end=' ')
            if len(model_correlations) > 1:
                print(f'(p-value: {model_correlations[1]:.9f})')


def plot_distance_to_gt(model_basename, distances_to_embeddings, sim_threshold, gt_threshold,
                        save_path, error_bars=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    title = f'Distance to ground truth embeddings ({model_basename}) (lower is better)'
    diff_df, se_df = pd.DataFrame(), pd.DataFrame()
    for model_name in distances_to_embeddings:
        model_distances = distances_to_embeddings[model_name]
        model_distances[['sim']] = utils.apply_threshold(model_distances[['sim']], sim_threshold)
        model_distances[['sim_gt']] = utils.apply_threshold(model_distances[['sim_gt']], gt_threshold)
        model_distances['diff'] = abs(model_distances['sim'] - model_distances['sim_gt'])
        mean_diff = model_distances.groupby('in_stimuli')['diff'].mean()
        std_diff = model_distances.groupby('in_stimuli')['diff'].std()
        se_diff = std_diff / np.sqrt(model_distances.shape[0])
        diff_df = pd.concat([diff_df, mean_diff.to_frame(model_name)], axis=1)
        se_df = pd.concat([se_df, se_diff.to_frame(model_name)], axis=1)
    if error_bars:
        diff_df.plot.bar(xlabel='Words present in SWOW-RP', yerr=se_df, capsize=4, ax=ax)
    else:
        diff_df.plot.bar(xlabel='Words present in SWOW-RP', ax=ax)
    ax.set_title(title)
    ax.legend()
    ax.set_ylabel('Similarity difference with SWOW-RP embeddings')
    plt.savefig(save_path / f'{title}.png')
    plt.show()


def plot_freq_to_sim(basename, similarities_to_answers, save_path, min_appearances):
    fig, ax = plt.subplots(figsize=(15, 6))
    title = f'Human frequency to model similarity ({basename})'
    for model_name in similarities_to_answers:
        model_sim_to_answers = similarities_to_answers[model_name]
        wa_freq_sim_to_plot = utils.filter_low_frequency_answers(model_sim_to_answers, min_appearances)
        ax.scatter(wa_freq_sim_to_plot['similarity'], wa_freq_sim_to_plot['freq'], label=model_name)
    ax.set_xlabel('Model similarity')
    ax.set_ylabel('Human frequency of answer')
    ax.set_title(title)
    ax.legend()
    plt.savefig(save_path / 'freq_to_sim.png')
    plt.show()


def plot_similarity(model_basename, similarities_to_subjs, sim_threshold, save_path, sort_by='texts', error_bars=True):
    if 'baseline' not in similarities_to_subjs:
        print('No baseline model found. Skipping similarity plots')
        return
    for axis, comparable in zip([0, 1], ['subjects', 'cues']):
        fig, ax = plt.subplots(figsize=(25, 15))
        title = f'Avg. similarity to {comparable} answers (baseline: {model_basename}) (higher is better)'
        print(f'\n------{title}------')
        mean_similarities, se_similarities = pd.DataFrame(), pd.DataFrame()
        for model_name in similarities_to_subjs:
            model_sim_to_subjs = utils.apply_threshold(similarities_to_subjs[model_name], sim_threshold)
            mean_subj_sim, se_subj_sim = report_similarity(model_name, model_sim_to_subjs, axis)
            mean_similarities = pd.concat([mean_similarities, mean_subj_sim], axis=1)
            se_similarities = pd.concat([se_similarities, se_subj_sim], axis=1)
        mean_similarities, se_similarities = compare_to_baseline(mean_similarities, se_similarities)
        mean_similarities = mean_similarities.sort_values(by=sort_by, ascending=False)
        se_similarities = se_similarities.reindex(mean_similarities.index)
        if error_bars:
            mean_similarities.plot.bar(yerr=se_similarities, capsize=4, ax=ax)
        else:
            mean_similarities.plot.bar(ax=ax)
        ax.set_title(title, fontsize=15)
        ax.legend(fontsize=15)
        ax.set_ylabel('Similarity diff. to baseline', fontsize=15)
        plt.savefig(save_path / f'{title}.png')
        plt.show()


def compare_to_baseline(mean_similarities, se_similarities):
    baseline_mean, baseline_se = mean_similarities['baseline'], se_similarities['baseline']
    mean_similarities = mean_similarities.drop(columns=['baseline'])
    se_similarities = se_similarities.drop(columns=['baseline'])
    mean_similarities = mean_similarities.subtract(baseline_mean, axis=0)
    se_similarities = se_similarities.add(baseline_se, axis=0)
    return mean_similarities, se_similarities


def report_similarity(model, similarities_df, axis):
    mean_subj_similarity = similarities_df.mean(axis=axis)
    std_subj_similarity = similarities_df.std(axis=axis)
    se_subj_similarity = std_subj_similarity / np.sqrt(similarities_df.shape[1])
    print(f'{model} mean: {mean_subj_similarity.mean():.4f} (std: {std_subj_similarity.mean():.4f})')
    return mean_subj_similarity.to_frame(model), se_subj_similarity.to_frame(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Model base name')
    parser.add_argument('-m', '--models', type=str, default='models',
                        help='Path to the trained models')
    parser.add_argument('-f', '--fraction', type=float, default=1.0,
                        help='Fraction of baseline corpus employed for model training')
    parser.add_argument('-ws', '--words_samples', type=int, default=8000,
                        help='Number of words to be sampled from the words association file for evaluation')
    parser.add_argument('-mf', '--min_freq', type=int, default=25,
                        help='Minimum number of occurrences for an answer in the words association file for evaluation')
    parser.add_argument('-t', '--threshold', type=float, default=0.02,
                        help='Threshold for the similarity values to be considered correct')
    parser.add_argument('-gt', '--gt_threshold', type=float, default=-0.08,
                        help='Threshold for the ground truth embeddings similarity values to be considered correct')
    parser.add_argument('-s', '--stimuli', type=str, default='stimuli',
                        help='Path to item files employed in the experiment')
    parser.add_argument('-e', '--embeddings', type=str, default='evaluation/SWOWRP_embeddings.vec',
                        help='Human derived word embeddings to be used as ground truth for evaluation')
    parser.add_argument('-sa', '--subjs_associations', type=str, default='evaluation/subjects_associations.csv',
                        help='Subjects free associations to words file to be employed for evaluation')
    parser.add_argument('-wa', '--words_associations', type=str, default='evaluation/SWOWRP_words_associations.csv',
                        help='Words associations file to be employed for evaluation')
    parser.add_argument('-wf', '--words_frequency', type=str, default='evaluation/wordsfreq.csv',
                        help='File containing the frequency of each word in the words associations file')
    parser.add_argument('-ss', '--sort_sim_by', type=str, default='texts',
                        help='Sort similarity plots by the specified model values')
    parser.add_argument('-se', '--standard_error', action='store_false', help='Plot error bars in similarity plots')
    parser.add_argument('-o', '--output', type=str, default='results', help='Where to save test results')
    args = parser.parse_args()
    sa_file, wa_file, wf_file = Path(args.subjs_associations), Path(args.words_associations), Path(args.words_frequency)
    output, stimuli_path, gt_embeddings_file = Path(args.output), Path(args.stimuli), Path(args.embeddings)
    model_path = utils.get_model_path(args.models, args.model_name, args.fraction)

    test(model_path, wa_file, sa_file, wf_file, args.min_freq, args.words_samples, args.threshold, args.gt_threshold,
         gt_embeddings_file, stimuli_path, output, args.sort_sim_by, args.standard_error)
