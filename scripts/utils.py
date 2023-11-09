import numpy as np

from scripts.corpora import Corpus


def get_words_in_corpus(stimuli_path):
    stimuli = Corpus(stimuli_path.name, 'local', 1.0, min_token_len=2, max_token_len=20, min_sentence_len=None)
    words_in_corpus = set()
    for sentence in stimuli.get_texts():
        for word in sentence['text']:
            words_in_corpus.add(word)
    return words_in_corpus


def subsample(series, n, seed):
    return series.sample(n, random_state=seed).to_numpy() if len(series) > n else series.to_numpy()


def filter_low_frequency_answers(words_answers_pairs, min_appearances):
    return words_answers_pairs[words_answers_pairs['n'] >= min_appearances]


def similarities(words_vectors, words, answers):
    for i, answer in enumerate(answers):
        answers[i] = word_similarity(words_vectors, words[i], answer)
    return answers


def word_similarity(words_vectors, word, answer):
    if answer is None or answer not in words_vectors or word not in words_vectors:
        return np.nan
    return words_vectors.similarity(word, answer)
