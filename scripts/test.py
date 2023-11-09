import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors
from scripts.utils import get_words_in_corpus, subsample, filter_low_frequency_answers, similarities


def test(model_path, wa_file, sa_file, stimuli_path, gt_embeddings_file, save_path, error_bars=True):
    models = [dir_ for dir_ in model_path.iterdir() if dir_.is_dir()]
    if len(models) == 0:
        raise ValueError(f'There are no models in {model_path}')
    if not (sa_file.exists() or wa_file.exists()):
        raise ValueError(f'Evaluation files missing: {wa_file} and {sa_file} do not exist')
    if not stimuli_path.exists():
        raise ValueError(f'Stimuli files missing: {stimuli_path} does not exist')
    subjs_associations = pd.read_csv(sa_file, index_col=0)
    words_associations = pd.read_csv(wa_file)
    gt_embeddings = KeyedVectors.load_word2vec_format(str(gt_embeddings_file))
    words_in_stimuli = get_words_in_corpus(stimuli_path)
    models_results = {model.name: {'similarity_to_subjs': None, 'similarity_to_answers': None, 'word_pairs': None,
                                   'distance_to_embeddings': None} for model in models}
    for model_dir in models:
        test_model(model_dir, words_associations, subjs_associations, gt_embeddings, words_in_stimuli, models_results)

    model_basename = model_path.name
    save_path = save_path / model_basename
    save_path.mkdir(exist_ok=True, parents=True)
    plot_similarity(model_basename, models_results, save_path, error_bars)
    plot_freq_to_sim(model_basename, models_results, save_path, min_appearances=5)
    plot_distance_to_gt_embeddings(model_basename, models_results, save_path, error_bars)
    print_words_pairs_correlations(models_results)


def test_model(model_dir, words_associations, subjs_associations, gt_embeddings, words_in_stimuli, models_results):
    model_file = model_dir / f'{model_dir.name}.model'
    model = Word2Vec.load(str(model_file))
    answers_sim = similarities(model.wv, words_associations['cue'], words_associations['answer'])
    wa_model_sim = words_associations.copy()
    wa_model_sim['similarity'] = answers_sim
    words = words_associations['cue'].drop_duplicates()
    distance_to_gt_embeddings = get_distance(model.wv, words, words_in_stimuli, gt_embeddings)
    sa_subj_sim = subjs_associations.copy().apply(lambda answers: similarities(model.wv, words, answers))
    models_results[model_dir.name]['similarity_to_subjs'] = sa_subj_sim
    models_results[model_dir.name]['similarity_to_answers'] = wa_model_sim
    models_results[model_dir.name]['word_pairs'] = evaluate_word_pairs(model.wv, wa_model_sim, model_dir)
    models_results[model_dir.name]['distance_to_embeddings'] = distance_to_gt_embeddings


def evaluate_word_pairs(words_vectors, freq_similarity_pairs, model_dir):
    temp_file = model_dir / 'word_pairs.csv'
    word_pairs = freq_similarity_pairs.drop(columns=['similarity', 'n'])
    word_pairs.to_csv(temp_file, index=False, header=False)
    pearson, spearman, oov_ratio = words_vectors.evaluate_word_pairs(temp_file, delimiter=',')
    temp_file.unlink()
    return pearson, spearman, [oov_ratio]


def get_distance(words_vectors, cues, words_in_stimuli, gt_embeddings, n=20):
    cues_in_stimuli, cues_off_stimuli = cues[cues.isin(words_in_stimuli)], cues[~cues.isin(words_in_stimuli)]
    cues_in_stimuli, cues_off_stimuli = subsample(cues_in_stimuli, n, seed=42), subsample(cues_off_stimuli, n, seed=42)

    cues_in_stimuli_sim = similarities(words_vectors, cues_in_stimuli, cues_in_stimuli)
    cues_in_stimuli_sim_gt = similarities(gt_embeddings, cues_in_stimuli, cues_in_stimuli)
    cues_off_stimuli_sim = similarities(words_vectors, cues_off_stimuli, cues_off_stimuli)
    cues_off_stimuli_sim_gt = similarities(gt_embeddings, cues_off_stimuli, cues_off_stimuli)

    diff_cues_in_stimuli = cues_in_stimuli_sim_gt - cues_in_stimuli_sim
    diff_cues_off_stimuli = cues_off_stimuli_sim_gt - cues_off_stimuli_sim
    df_stimuli = pd.DataFrame({'cue': cues_in_stimuli, 'diff': diff_cues_in_stimuli, 'in_stimuli': True})
    df_off_stimuli = pd.DataFrame({'cue': cues_off_stimuli, 'diff': diff_cues_off_stimuli, 'in_stimuli': False})
    return pd.concat([df_stimuli, df_off_stimuli])


def plot_distance_to_gt_embeddings(model_basename, models_results, save_path, error_bars=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    title = f'Distance to ground truth embeddings ({model_basename})'
    diff_df, se_df = pd.DataFrame(), pd.DataFrame()
    for model in models_results:
        model_results = models_results[model]['distance_to_embeddings']
        mean_diff = model_results.groupby('in_stimuli')['diff'].mean()
        std_diff = model_results.groupby('in_stimuli')['diff'].std()
        se_diff = std_diff / np.sqrt(model_results.shape[0])
        diff_df = pd.concat([diff_df, mean_diff.to_frame(model)], axis=1)
        se_df = pd.concat([se_df, se_diff.to_frame(model)], axis=1)
    if error_bars:
        diff_df.plot.bar(yerr=se_df, capsize=4, ax=ax)
    else:
        diff_df.plot.bar(xlabel='Words present in stimuli', ax=ax)
    ax.set_title(title)
    ax.legend()
    ax.set_ylabel('Similarity difference with SWOW-RP embeddings')
    plt.savefig(save_path / f'{title}.png')
    plt.show()


def plot_similarity(model_basename, models_results, save_path, error_bars=True):
    for axis, title in zip([0, 1], ['subjects', 'cues']):
        fig, ax = plt.subplots(figsize=(25, 15))
        title = f'Avg. similarity to {title} answers ({model_basename})'
        print(f'\n------{title}------')
        mean_similarities, se_similarities = pd.DataFrame(), pd.DataFrame()
        for model in models_results:
            model_results = models_results[model]
            mean_subj_sim, se_subj_sim = report_similarity(model, model_results['similarity_to_subjs'], axis)
            mean_similarities = pd.concat([mean_similarities, mean_subj_sim], axis=1)
            se_similarities = pd.concat([se_similarities, se_subj_sim], axis=1)
        if error_bars:
            mean_similarities.plot.bar(yerr=se_similarities, capsize=4, ax=ax)
        else:
            mean_similarities.plot.bar(ax=ax)
        ax.set_title(title, fontsize=15)
        ax.legend(fontsize=15)
        ax.set_ylabel('Similarity', fontsize=15)
        plt.savefig(save_path / f'{title}.png')
        plt.show()


def plot_freq_to_sim(basename, models_results, save_path, min_appearances):
    fig, ax = plt.subplots(figsize=(15, 6))
    title = f'Human frequency to model similarity ({basename})'
    for model in models_results:
        model_results = models_results[model]['similarity_to_answers']
        wa_freq_sim_to_plot = filter_low_frequency_answers(model_results, min_appearances)
        ax.scatter(wa_freq_sim_to_plot['similarity'], wa_freq_sim_to_plot['freq'], label=model)
    ax.set_xlabel('Model similarity')
    ax.set_ylabel('Human frequency of answer')
    ax.set_title(title)
    ax.legend()
    plt.savefig(save_path / 'freq_to_sim.png')
    plt.show()


def report_similarity(model, similarities_df, axis):
    mean_subj_similarity = similarities_df.mean(axis=axis)
    std_subj_similarity = similarities_df.std(axis=axis)
    se_subj_similarity = std_subj_similarity / np.sqrt(similarities_df.shape[1])
    print(f'{model} mean: {round(mean_subj_similarity.mean(), 4)} (std: {round(std_subj_similarity.mean(), 4)})')
    return mean_subj_similarity.to_frame(model), se_subj_similarity.to_frame(model)


def print_words_pairs_correlations(models_results):
    print('\n---Correlation between similarity and frequency of response for cue-answer pairs---')
    measures = ['Pearson', 'Spearman rank-order', 'Out of vocabulary ratio']
    for i, measure in enumerate(measures):
        print(f'{measure}')
        for model in models_results:
            model_results = models_results[model]['word_pairs'][i]
            print(f'{model}: {round(model_results[0], 4)}', end=' ')
            if len(model_results) > 1:
                print(f'(p-value: {round(model_results[1], 4)})')
