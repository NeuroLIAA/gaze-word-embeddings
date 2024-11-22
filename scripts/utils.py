from pathlib import Path
import numpy as np
from scripts.corpora import Corpus


def get_words_in_corpus(stimuli_path):
    stimuli = Corpus(stimuli_path.name, 'local', 1.0, min_token_len=2, max_token_len=20,
                     min_sentence_len=4, max_sentence_len=40)
    words_in_corpus = set()
    for sentence in stimuli.data:
        for word in sentence['text']:
            words_in_corpus.add(word)
    return words_in_corpus


def subsample(series, n, seed):
    return series.sample(n, random_state=seed) if len(series) > n else series


def in_off_stimuli_word_pairs(words_in_stimuli, words_associations, words_frequency, n, resamples, rng, seed=42):
    words_frequency = words_frequency.rename(columns={'word': 'cue', 'log_cnt': 'cue_log_cnt'})
    words_pairs = words_associations.merge(words_frequency, on='cue', how='left')
    in_stimuli = words_pairs[(words_pairs['cue'].isin(words_in_stimuli))
                             & (words_pairs['answer'].isin(words_in_stimuli))]
    off_stimuli = words_pairs[(~words_pairs['cue'].isin(words_in_stimuli))
                              & (~words_pairs['answer'].isin(words_in_stimuli))]
    in_stimuli_word_freq = in_stimuli['cue_log_cnt']
    matched_words_names, off_stimuli_cp = [], off_stimuli.copy()
    for log_cnt in in_stimuli_word_freq:
        matched_words = (off_stimuli_cp['cue_log_cnt'] - log_cnt).abs().argsort()[:1]
        matched_word_name = off_stimuli_cp.iloc[matched_words.sample(random_state=seed).iloc[0]].name
        matched_words_names.append(matched_word_name)
        off_stimuli_cp.drop(matched_word_name, inplace=True)
    off_stimuli = off_stimuli[off_stimuli.index.isin(matched_words_names)]

    seeds = rng.integers(0, 10000, size=resamples)
    in_stimuli_wp, off_stimuli_wp = [], []
    for seed in seeds:
        in_stimuli_wp.append(subsample(in_stimuli, n, seed))
        off_stimuli_wp.append(subsample(off_stimuli, n, seed))

    return in_stimuli_wp, off_stimuli_wp


def filter_low_frequency_answers(words_answers_pairs, min_appearances):
    return words_answers_pairs[words_answers_pairs['n'] >= min_appearances]


def similarities(words_vectors, words, answers):
    similarities = np.zeros(len(answers))
    for i, word_pair in enumerate(zip(words, answers)):
        word, answer = word_pair
        similarities[i] = word_similarity(words_vectors, word, answer)
    return similarities


def word_similarity(words_vectors, word, answer):
    if answer is None or answer not in words_vectors or word not in words_vectors:
        return np.nan
    return words_vectors.similarity(word, answer)


def embeddings(words_vectors, words):
    embeddings, corresponding_words = [], []
    for word in words:
        if word in words_vectors:
            embeddings.append(words_vectors[word])
            corresponding_words.append(word)
    return np.array(embeddings), corresponding_words


def get_embeddings_path(embeddings, data_name, fraction):
    if fraction < 1.0:
        embeddings_path = Path(embeddings) / f'{data_name}_{int(fraction * 100)}%'
    else:
        embeddings_path = Path(embeddings) / data_name
    return embeddings_path
