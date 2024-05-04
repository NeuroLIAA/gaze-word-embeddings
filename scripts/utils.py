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
    words = words_associations['cue'].drop_duplicates()
    words = words_frequency[words_frequency['word'].isin(words)]
    in_stimuli, off_stimuli = words[words['word'].isin(words_in_stimuli)], words[~words['word'].isin(words_in_stimuli)]
    in_stimuli = subsample(in_stimuli, n, seed)
    matched_words = []
    for log_cnt in in_stimuli['log_cnt']:
        matched_word = off_stimuli.iloc[(off_stimuli['log_cnt'] - log_cnt).abs().argsort()[:1]]
        matched_words.append(matched_word['word'].sample(1, random_state=seed).values[0])
        off_stimuli = off_stimuli[off_stimuli['word'] != matched_words[-1]]
    off_stimuli = words[words['word'].isin(matched_words)]
    in_stimuli, off_stimuli = build_all_pairs(in_stimuli['word']), build_all_pairs(off_stimuli['word'])
    in_stimuli['in_stimuli'], off_stimuli['in_stimuli'] = True, False
    word_pairs = pd.concat([in_stimuli, off_stimuli])
    word_pairs = word_pairs[word_pairs['cue'] != word_pairs['answer']]

    return word_pairs, in_stimuli['cue'].unique(), off_stimuli['cue'].unique()


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


def get_model_path(models, model_name, fraction):
    if fraction < 1.0:
        model_path = Path(models) / f'{model_name}_{int(fraction * 100)}%'
    else:
        model_path = Path(models) / model_name
    return model_path
