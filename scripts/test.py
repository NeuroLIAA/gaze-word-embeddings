import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from gensim.models import Word2Vec, KeyedVectors
from scripts.utils import get_words_in_corpus, subsample, filter_low_frequency_answers, similarities


def test(model_path, wa_file, sa_file, min_freq, num_samples, sim_threshold, stimuli_path, gt_embeddings_file,
         save_path, sort_sim_by, error_bars):
    models = [dir_ for dir_ in model_path.iterdir() if dir_.is_dir()]
    if len(models) == 0:
        raise ValueError(f'There are no models in {model_path}')
    if not (sa_file.exists() or wa_file.exists()):
        raise ValueError(f'Evaluation files missing: {wa_file} and {sa_file} do not exist')
    if not stimuli_path.exists():
        raise ValueError(f'Stimuli files missing: {stimuli_path} does not exist')
    subjs_associations = pd.read_csv(sa_file, index_col=0)
    words_associations = pd.read_csv(wa_file)
    words_associations = words_associations[words_associations['cue'].isin(
        subsample(words_associations['cue'], num_samples, seed=42))]
    gt_embeddings = KeyedVectors.load_word2vec_format(str(gt_embeddings_file))
    words_in_stimuli = get_words_in_corpus(stimuli_path)
    models_results = {'similarity_to_subjs': {}, 'similarity_to_answers': {}, 'word_pairs': {},
                      'distance_to_embeddings': {}}
    for model_dir in models:
        model_wv = Word2Vec.load(str(model_dir / f'{model_dir.name}.model')).wv
        test_model(model_wv, model_dir.name, words_associations, subjs_associations, gt_embeddings, words_in_stimuli,
                   models_results)

    model_basename = model_path.name
    save_path = save_path / model_basename
    save_path.mkdir(exist_ok=True, parents=True)
    plot_similarity(model_basename, models_results['similarity_to_subjs'], sim_threshold,
                    save_path, sort_sim_by, error_bars)
    plot_freq_to_sim(model_basename, models_results['similarity_to_answers'], save_path, min_appearances=min_freq)
    plot_distance_to_gt_embeddings(model_basename, models_results['distance_to_embeddings'], save_path, error_bars)
    print_words_pairs_correlations(models_results['word_pairs'])


def test_model(model_wv, model_name, words_associations, subjs_associations, gt_embeddings, words_in_stimuli,
               models_results):
    answers_sim = similarities(model_wv, words_associations['cue'], words_associations['answer'])
    wa_model_sim = words_associations.copy()
    wa_model_sim['similarity'] = answers_sim
    words = words_associations['cue'].drop_duplicates()
    distance_to_gt_embeddings = get_distance(model_wv, words, words_in_stimuli, gt_embeddings)
    cues = subjs_associations.index
    sa_subj_sim = subjs_associations.copy().apply(lambda answers: similarities(model_wv, cues, answers))
    models_results['similarity_to_subjs'][model_name] = sa_subj_sim
    models_results['similarity_to_answers'][model_name] = wa_model_sim
    models_results['word_pairs'][model_name] = evaluate_word_pairs(model_wv, wa_model_sim)
    models_results['distance_to_embeddings'][model_name] = distance_to_gt_embeddings


def get_distance(words_vectors, cues, words_in_stimuli, gt_embeddings, n=20):
    cues_in_stimuli, cues_off_stimuli = cues[cues.isin(words_in_stimuli)], cues[~cues.isin(words_in_stimuli)]
    cues_in_stimuli, cues_off_stimuli = subsample(cues_in_stimuli, n, seed=42), subsample(cues_off_stimuli, n, seed=42)

    # TODO: do not save diff, save similarity
    cues_in_stimuli_sim = similarities(words_vectors, cues_in_stimuli, cues_in_stimuli)
    cues_in_stimuli_sim_gt = similarities(gt_embeddings, cues_in_stimuli, cues_in_stimuli)
    cues_off_stimuli_sim = similarities(words_vectors, cues_off_stimuli, cues_off_stimuli)
    cues_off_stimuli_sim_gt = similarities(gt_embeddings, cues_off_stimuli, cues_off_stimuli)

    diff_cues_in_stimuli = cues_in_stimuli_sim_gt - cues_in_stimuli_sim
    diff_cues_off_stimuli = cues_off_stimuli_sim_gt - cues_off_stimuli_sim
    df_stimuli = pd.DataFrame({'cue': cues_in_stimuli, 'diff': diff_cues_in_stimuli, 'in_stimuli': True})
    df_off_stimuli = pd.DataFrame({'cue': cues_off_stimuli, 'diff': diff_cues_off_stimuli, 'in_stimuli': False})
    return pd.concat([df_stimuli, df_off_stimuli])


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


def plot_distance_to_gt_embeddings(model_basename, distances_to_embeddings, save_path, error_bars=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    title = f'Distance to ground truth embeddings ({model_basename})'
    diff_df, se_df = pd.DataFrame(), pd.DataFrame()
    for model_name in distances_to_embeddings:
        model_distances = distances_to_embeddings[model_name]
        mean_diff = model_distances.groupby('in_stimuli')['diff'].mean()
        std_diff = model_distances.groupby('in_stimuli')['diff'].std()
        se_diff = std_diff / np.sqrt(model_distances.shape[0])
        diff_df = pd.concat([diff_df, mean_diff.to_frame(model_name)], axis=1)
        se_df = pd.concat([se_df, se_diff.to_frame(model_name)], axis=1)
    if error_bars:
        diff_df.plot.bar(yerr=se_df, capsize=4, ax=ax)
    else:
        diff_df.plot.bar(xlabel='Words present in stimuli', ax=ax)
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
        wa_freq_sim_to_plot = filter_low_frequency_answers(model_sim_to_answers, min_appearances)
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
        title = f'Avg. similarity to {comparable} answers (baseline: {model_basename})'
        print(f'\n------{title}------')
        mean_similarities, se_similarities = pd.DataFrame(), pd.DataFrame()
        for model_name in similarities_to_subjs:
            model_sim_to_subjs = similarities_to_subjs[model_name]
            model_sim_to_subjs = model_sim_to_subjs.applymap(lambda sim: 0 if sim < sim_threshold else 1)
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
