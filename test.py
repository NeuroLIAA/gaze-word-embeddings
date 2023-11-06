import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec


def test(model_path, wa_file, save_path, error_bars=True):
    models = [dir_ for dir_ in model_path.iterdir() if dir_.is_dir()]
    if len(models) == 0:
        raise ValueError(f'There are no models in {model_path}')
    if not wa_file.exists():
        raise ValueError(f'Words association file {wa_file} does not exist')
    words_associations = pd.read_csv(wa_file, index_col=0)
    words = words_associations.index
    models_results = {model.name: {'similarity_to_subjs': None, 'similarity_to_answers': None, 'word_pairs': None}
                      for model in models}
    for model_dir in models:
        test_model(model_dir, words, words_associations, models_results, save_path)

    model_basename = model_path.name
    save_path = save_path / model_basename
    save_path.mkdir(exist_ok=True, parents=True)
    plot_similarity(model_basename, models_results, save_path, error_bars)
    plot_freq_to_sim(model_basename, models_results, words_associations, save_path, min_appearences=5)
    print_words_pairs_correlations(models_results)


def test_model(model_dir, words, words_associations, models_results, save_path):
    model_file = model_dir / f'{model_dir.name}.model'
    model = Word2Vec.load(str(model_file))
    wa_freq_sim_df = wa_similarities(answers_frequency(words_associations), model)
    wa_subj_sim_df = words_associations.copy().apply(lambda answers: similarities(model, words, answers))
    models_results[model_dir.name]['similarity_to_subjs'] = wa_subj_sim_df
    models_results[model_dir.name]['similarity_to_answers'] = wa_freq_sim_df
    models_results[model_dir.name]['word_pairs'] = evaluate_word_pairs(model, wa_freq_sim_df, save_path)


def filter_low_frequency_answers(words_answers_pairs, words_associations, min_appearances):
    num_answers = answers_frequency(words_associations, normalized=False)
    return words_answers_pairs[words_answers_pairs.apply(
        lambda row: num_answers[row['cue']][row['answer']] >= min_appearances, axis=1)]


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


def plot_freq_to_sim(basename, models_results, words_associations, save_path, min_appearences):
    fig, ax = plt.subplots(figsize=(15, 6))
    title = f'Human frequency to model similarity ({basename})'
    for model in models_results:
        model_results = models_results[model]['similarity_to_answers']
        wa_freq_sim_to_plot = filter_low_frequency_answers(model_results, words_associations, min_appearences)
        wa_freq_sim_to_plot.plot.scatter(x='similarity', y='freq', figsize=(15, 5), ax=ax, label=model)
    ax.set_xlabel('Model similarity')
    ax.set_ylabel('Human frequency of answer')
    ax.set_title(title)
    ax.legend()
    plt.savefig(save_path / 'freq_to_sim.png')
    plt.show()


def wa_similarities(freq, model):
    words_pairs = []
    for cue in freq:
        cue_answers = freq[cue]
        for answer in cue_answers.keys():
            words_pairs.append((cue, answer, cue_answers[answer], word_similarity(model, cue, answer)))

    words_pairs = pd.DataFrame(words_pairs, columns=['cue', 'answer', 'freq', 'similarity'])
    return words_pairs


def similarities(model, words, answers):
    for i, answer in enumerate(answers):
        answers[i] = word_similarity(model, words[i], answer)
    return answers


def word_similarity(model, word, answer):
    if answer is None or answer not in model.wv or word not in model.wv:
        return np.nan
    return model.wv.similarity(word, answer)


def answers_frequency(words_associations, normalized=True):
    return {word: words_associations.loc[word].value_counts(normalize=normalized) for word in words_associations.index}
