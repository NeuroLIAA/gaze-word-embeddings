import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec


def similarity_to_subjs(similarities_df):
    mean_subj_similarity = similarities_df.mean()
    std_subj_similarity = similarities_df.std()
    se_subj_similarity = std_subj_similarity / np.sqrt(similarities_df.shape[1])
    mean_subj_similarity.plot.bar(yerr=se_subj_similarity, capsize=4, figsize=(20, 10), fontsize=20)
    plt.ylabel('Similarity', fontsize=20)
    plt.show()
    print('------Average similarity to subjects answers------')
    print(f'Mean: {mean_subj_similarity.mean()} (std: {std_subj_similarity.mean()})')


def test(model_path, wa_file):
    model_file = model_path / f'{model_path.name}.model'
    if not model_file.exists():
        raise ValueError('The specified models does not exist')
    if not wa_file.exists():
        raise ValueError('The specified words association file does not exist')
    model = Word2Vec.load(str(model_file))
    words_associations = pd.read_csv(wa_file, index_col=0)
    words = words_associations.index
    frequency = answers_frequency(words_associations)
    freq_similarity_pairs = add_model_similarity(frequency, model)
    similarities_df = words_associations.apply(lambda answers: similarities(model, words, answers))
    save_path = model_path / 'test'
    save_path.mkdir(exist_ok=True)
    similarities_df.to_csv(save_path / f'{wa_file.stem}_similarity.csv')
    freq_similarity_pairs.to_csv(save_path / f'{wa_file.stem}_freq.csv', index=False)
    similarity_to_subjs(similarities_df)
    evaluate_word_pairs(model, freq_similarity_pairs, save_path)
    return similarities_df


def evaluate_word_pairs(model, freq_similarity_pairs, save_path):
    filename = save_path / 'word_pairs.csv'
    word_pairs = freq_similarity_pairs.drop(columns=['similarity'])
    word_pairs.to_csv(filename, index=False, header=False)
    pearson, spearman, oov_ratio = model.wv.evaluate_word_pairs(filename, delimiter=',')
    print('------Correlation between similarity and frequency of response for cue-answer pairs------')
    print(f'Pearson correlation coefficient: {pearson[0]} (p-value: {pearson[1]})')
    print(f'Spearman rank-order correlation coefficient: {spearman.correlation} (p-value: {spearman.pvalue})')
    print(f'Out of vocabulary ratio: {oov_ratio}')
    filename.unlink()


def add_model_similarity(freq, model):
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


def answers_frequency(words_associations):
    return {word: words_associations.loc[word].value_counts(normalize=True) for word in words_associations.index}
