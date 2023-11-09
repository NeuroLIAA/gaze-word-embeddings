import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors
from corpora import Corpus


def test(model_path, wa_file, sa_file, stimuli_path, gt_embeddings_file, save_path, error_bars=True):
    models = [dir_ for dir_ in model_path.iterdir() if dir_.is_dir()]
    if len(models) == 0:
        raise ValueError(f'There are no models in {model_path}')
    if not (sa_file.exists() and wa_file.exists()):
        raise ValueError(f'Evaluation file(s) missing: {wa_file} and {sa_file} do not exist')
    if not stimuli_path.exists():
        raise ValueError(f'Stimuli files missing: {stimuli_path} does not exist')
    subjs_associations = pd.read_csv(sa_file, index_col=0) if sa_file.exists() else None
    words_associations = pd.read_csv(wa_file) if wa_file.exists() else None
    gt_embeddings = KeyedVectors.load_word2vec_format(str(gt_embeddings_file)) if gt_embeddings_file.exists() else None
    words_in_stimuli = get_words_in_corpus(stimuli_path)
    models_results = {model.name: {'similarity_to_subjs': None, 'similarity_to_answers': None, 'word_pairs': None,
                                   'distance_to_embeddings': None} for model in models}
    for model_dir in models:
        test_model(model_dir, words_associations, subjs_associations, gt_embeddings, words_in_stimuli,
                   models_results, save_path)

    model_basename = model_path.name
    save_path = save_path / model_basename
    save_path.mkdir(exist_ok=True, parents=True)
    plot_similarity(model_basename, models_results, save_path, error_bars)
    plot_freq_to_sim(model_basename, models_results, subjs_associations, save_path, min_appearences=5)
    plot_distance_to_gt_embeddings(model_basename, models_results, save_path)
    print_words_pairs_correlations(models_results)


def get_words_in_corpus(stimuli_path):
    stimuli = Corpus(stimuli_path.name, 'local', 1.0, min_token_len=2, max_token_len=20, min_sentence_len=None)
    words_in_corpus = set()
    for sentence in stimuli.get_texts():
        for word in sentence['text']:
            words_in_corpus.add(word)
    return words_in_corpus


def test_model(model_dir, words_associations, subjs_associations, gt_embeddings, words_in_stimuli,
               models_results, save_path):
    model_file = model_dir / f'{model_dir.name}.model'
    model = Word2Vec.load(str(model_file))
    answers_sim = similarities(model.wv, words_associations['cue'], words_associations['answer'].to_list())
    words_associations['similarity'] = answers_sim
    wa_model_sim_df = words_associations.copy()
    distance_to_gt_embeddings = get_distance(model.wv, words_associations['cue'], words_in_stimuli, gt_embeddings)
    words = subjs_associations.index
    sa_subj_sim_df = subjs_associations.copy().apply(lambda answers: similarities(model.wv, words, answers))
    models_results[model_dir.name]['similarity_to_subjs'] = sa_subj_sim_df
    models_results[model_dir.name]['similarity_to_answers'] = wa_model_sim_df
    models_results[model_dir.name]['word_pairs'] = evaluate_word_pairs(model, wa_model_sim_df, save_path)
    models_results[model_dir.name]['distance_to_embeddings'] = distance_to_gt_embeddings


def get_distance(words_vectors, cues, words_in_stimuli, gt_embeddings, n=20):
    cues_in_stimuli = cues[cues.isin(words_in_stimuli)]
    cues_out_stimuli = cues[~cues.isin(words_in_stimuli)]
    cues_in_stimuli = cues_in_stimuli.sample(n, random_state=42) if len(cues_in_stimuli) > n else cues_in_stimuli
    cues_out_stimuli = cues_out_stimuli.sample(n, random_state=42) if len(cues_out_stimuli) > n else cues_out_stimuli
    cues_in_stimuli_sim = similarities(words_vectors, cues_in_stimuli, cues_in_stimuli.to_list())
    cues_out_stimuli_sim = similarities(words_vectors, cues_out_stimuli, cues_out_stimuli.to_list())
    cues_in_stimuli_sim_gt = similarities(gt_embeddings, cues_in_stimuli, cues_in_stimuli.to_list())
    cues_out_stimuli_sim_gt = similarities(gt_embeddings, cues_out_stimuli, cues_out_stimuli.to_list())
    diff_cues_in_stimuli = cues_in_stimuli_sim_gt - cues_in_stimuli_sim
    diff_cues_out_stimuli = cues_out_stimuli_sim_gt - cues_out_stimuli_sim
    df_stimuli = pd.DataFrame({'cue': cues_in_stimuli, 'diff': diff_cues_in_stimuli, 'in_stimuli': True})
    df_out_stimuli = pd.DataFrame({'cue': cues_out_stimuli, 'diff': diff_cues_out_stimuli, 'in_stimuli': False})
    return pd.concat([df_stimuli, df_out_stimuli])


def plot_distance_to_gt_embeddings(model_basename, models_results, save_path):
    pass


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


def report_similarity(model, similarities_df, axis):
    mean_subj_similarity = similarities_df.mean(axis=axis)
    std_subj_similarity = similarities_df.std(axis=axis)
    se_subj_similarity = std_subj_similarity / np.sqrt(similarities_df.shape[1])
    print(f'{model} mean: {round(mean_subj_similarity.mean(), 4)} (std: {round(std_subj_similarity.mean(), 4)})')
    return mean_subj_similarity.to_frame(model), se_subj_similarity.to_frame(model)


def evaluate_word_pairs(model, freq_similarity_pairs, save_path):
    temp_file = save_path / 'word_pairs.csv'
    word_pairs = freq_similarity_pairs.drop(columns=['similarity'])
    word_pairs.to_csv(temp_file, index=False, header=False)
    pearson, spearman, oov_ratio = model.wv.evaluate_word_pairs(temp_file, delimiter=',')
    temp_file.unlink()
    return pearson, spearman, [oov_ratio]


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


def plot_freq_to_sim(basename, models_results, subjs_associations, save_path, min_appearences):
    fig, ax = plt.subplots(figsize=(15, 6))
    title = f'Human frequency to model similarity ({basename})'
    for model in models_results:
        model_results = models_results[model]['similarity_to_answers']
        wa_freq_sim_to_plot = filter_low_frequency_answers(model_results, subjs_associations, min_appearences)
        ax.scatter(wa_freq_sim_to_plot['similarity'], wa_freq_sim_to_plot['freq'], label=model)
    ax.set_xlabel('Model similarity')
    ax.set_ylabel('Human frequency of answer')
    ax.set_title(title)
    ax.legend()
    plt.savefig(save_path / 'freq_to_sim.png')
    plt.show()


def filter_low_frequency_answers(words_answers_pairs, subjs_associations, min_appearances):
    num_answers = answers_frequency(subjs_associations, normalized=False)
    return words_answers_pairs[words_answers_pairs.apply(
        lambda row: num_answers[row['cue']][row['answer']] >= min_appearances, axis=1)]


def similarities(words_vectors, words, answers):
    for i, answer in enumerate(answers):
        answers[i] = word_similarity(words_vectors, words[i], answer)
    return answers


def word_similarity(words_vectors, word, answer):
    if answer is None or answer not in words_vectors or word not in words_vectors:
        return np.nan
    return words_vectors.similarity(word, answer)


def answers_frequency(words_associations, normalized=True):
    return {word: words_associations.loc[word].value_counts(normalize=normalized) for word in words_associations.index}
