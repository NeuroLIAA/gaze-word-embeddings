from pathlib import Path
import numpy as np
import pandas as pd
from scripts.corpora import Corpus


def get_words_in_corpus(stimuli_path):
    stimuli = Corpus(stimuli_path.name, 'local', 1.0, min_token_len=2, max_token_len=20, min_sentence_len=1)
    words_in_corpus = set()
    for sentence in stimuli.data:
        for word in sentence['text']:
            words_in_corpus.add(word)
    return words_in_corpus


def subsample(series, n, seed):
    return series.sample(n, random_state=seed) if len(series) > n else series


def in_off_stimuli_word_pairs(words_in_stimuli, words_associations, words_frequency, n=100, seed=42):
    words_frequency = words_frequency.rename(columns={'word': 'cue', 'log_cnt': 'cue_log_cnt'})
    words_pairs = words_associations.merge(words_frequency, on='cue', how='left')
    in_stimuli = words_pairs[(words_pairs['cue'].isin(words_in_stimuli))
                             & (words_pairs['answer'].isin(words_in_stimuli))]
    off_stimuli = words_pairs[(~words_pairs['cue'].isin(words_in_stimuli))
                              & (~words_pairs['answer'].isin(words_in_stimuli))]
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 10000, size=n)
    in_stimuli_wp, off_stimuli_wp = [], []
    for seed in seeds:
        in_stimuli_wp.append(subsample(in_stimuli, n, seed))
        matched_words = []
        for log_cnt in in_stimuli_wp[-1]['cue_log_cnt']:
            matched_word = off_stimuli.iloc[(off_stimuli['cue_log_cnt'] - log_cnt).abs().argsort()[:1]]
            matched_words.append(matched_word['cue'].sample(1, random_state=seed).values[0])
        off_stimuli_wp.append(off_stimuli[off_stimuli['cue'].isin(matched_words)])

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


def apply_threshold(similarity_df, threshold):
    return similarity_df.map(lambda x: 0 if x < threshold or np.isnan(x) else 1)


def build_all_pairs(words):
    words_pairs = pd.DataFrame({'cue': np.repeat(words, len(words)),
                               'answer': np.tile(words, len(words))})
    return words_pairs


def get_embeddings_path(models, model_name, fraction):
    if fraction < 1.0:
        model_path = Path(models) / f'{model_name}_{int(fraction * 100)}%'
    else:
        model_path = Path(models) / model_name
    return model_path
