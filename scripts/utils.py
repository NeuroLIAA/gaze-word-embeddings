from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from torch import nn as nn

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


def in_off_stimuli_word_pairs(words_with_measurements, words_in_stimuli, words_similarities, n, resamples, seed=42):
    in_stimuli = words_similarities[(words_similarities['word1'].isin(words_with_measurements))
                                    & (words_similarities['word2'].isin(words_with_measurements))]
    off_stimuli = words_similarities[(~words_similarities['word1'].isin(words_in_stimuli))
                                     & (~words_similarities['word2'].isin(words_in_stimuli))]
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 10000, size=resamples)
    in_stimuli_wp, off_stimuli_wp = [], []
    for seed in seeds:
        in_stimuli_wp.append(subsample(in_stimuli, n // 10, seed))
        off_stimuli_wp.append(subsample(off_stimuli, n // 10, seed))

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


def compute_fix_loss(fix_preds, fix_labels, fix_corrs, fix_pvalues, n_gaze_features):
    if n_gaze_features == 1:
        fix_preds = fix_preds.unsqueeze(dim=1)
    fix_loss = nn.functional.l1_loss(fix_preds, fix_labels)
    fix_preds = fix_preds.cpu().detach().numpy()
    fix_labels = fix_labels.cpu().detach().numpy()
    batch_correlations = [spearmanr(fix_preds[:, i], fix_labels[:, i], nan_policy='omit')
                          for i in range(n_gaze_features)]
    for i in range(n_gaze_features):
        fix_corrs[i].append(batch_correlations[i].correlation)
        fix_pvalues[i].append(batch_correlations[i].pvalue)
    return fix_loss


def print_batch_corrs(gaze_features, fix_corrs, fix_pvalues, n_gaze_features):
    if np.any(fix_corrs):
        for i in range(n_gaze_features):
            print(f'{gaze_features[i]} correlation: {np.nanmean(fix_corrs[i]):.3f} '
                  f'(+/- {np.nanstd(fix_corrs[i]):.3f}) | p-value: {np.nanmean(fix_pvalues[i]):.3f} '
                  f'(+/- {np.nanstd(fix_pvalues[i]):.3f})')
