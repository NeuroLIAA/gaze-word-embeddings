import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec


def test(model_path, wa_file):
    model_file = model_path / f'{model_path.name}.model'
    if not model_file.exists():
        raise ValueError(f'Model {model_file} does not exist')
    if not wa_file.exists():
        raise ValueError(f'Words association file {wa_file} does not exist')
    model = Word2Vec.load(str(model_file))
    words_associations = pd.read_csv(wa_file, index_col=0)
    words = words_associations.index
    wa_freq_sim_df = wa_similarities(answers_frequency(words_associations), model)
    wa_subj_sim_df = words_associations.copy().apply(lambda answers: similarities(model, words, answers))
    save_path = model_path / 'test'
    save_path.mkdir(exist_ok=True)
    wa_subj_sim_df.to_csv(save_path / f'{wa_file.stem}_similarity.csv')
    wa_freq_sim_df.to_csv(save_path / f'{wa_file.stem}_freq.csv', index=False)
    similarity_to_subjs(wa_subj_sim_df, save_path)
    similarity_to_cues(wa_subj_sim_df, save_path)
    evaluate_word_pairs(model, wa_freq_sim_df, save_path)
    plot_freq_to_sim(wa_freq_sim_df, words_associations, save_path, min_appearences=2)
    return wa_subj_sim_df


def filter_low_frequency_answers(words_answers_pairs, words_associations, min_appearances):
    num_answers = answers_frequency(words_associations, normalized=False)
    return words_answers_pairs[words_answers_pairs.apply(
        lambda row: num_answers[row['cue']][row['answer']] >= min_appearances, axis=1)]


def similarity_to_cues(similarities_df, save_path):
    report_similarity(similarities_df, 'Avg. similarity to cues answers', 1, save_path)


def similarity_to_subjs(similarities_df, save_path):
    report_similarity(similarities_df, 'Avg. similarity to subjects answers', 0, save_path)


def report_similarity(similarities_df, title, axis, save_path):
    mean_subj_similarity = similarities_df.mean(axis=axis)
    std_subj_similarity = similarities_df.std(axis=axis)
    se_subj_similarity = std_subj_similarity / np.sqrt(similarities_df.shape[1])
    mean_subj_similarity.plot.bar(yerr=se_subj_similarity, capsize=4, figsize=(20, 10), fontsize=20)
    plt.title(title, fontsize=20)
    plt.ylabel('Similarity', fontsize=20)
    plt.savefig(save_path / f'{title}.png')
    plt.show()
    print(f'------{title}------')
    print(f'Mean: {mean_subj_similarity.mean()} (std: {std_subj_similarity.mean()})\n')


def evaluate_word_pairs(model, freq_similarity_pairs, save_path):
    filename = save_path / 'word_pairs.csv'
    word_pairs = freq_similarity_pairs.drop(columns=['similarity'])
    word_pairs.to_csv(filename, index=False, header=False)
    pearson, spearman, oov_ratio = model.wv.evaluate_word_pairs(filename, delimiter=',')
    print('------Correlation between similarity and frequency of response for cue-answer pairs------')
    print(f'Pearson correlation coefficient: {pearson[0]} (p-value: {pearson[1]})')
    print(f'Spearman rank-order correlation coefficient: {spearman.correlation} (p-value: {spearman.pvalue})')
    print(f'Out of vocabulary ratio: {oov_ratio}\n')
    filename.unlink()


def plot_freq_to_sim(wa_freq_sim_df, words_associations, save_path, min_appearences):
    wa_freq_sim_to_plot = filter_low_frequency_answers(wa_freq_sim_df, words_associations, min_appearences)
    wa_freq_sim_to_plot.plot.scatter(x='similarity', y='freq', figsize=(15, 5),
                                     title='Human frequency to model similarity', xlabel='Model similarity',
                                     ylabel='Human frequency of answer')
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
