from pathlib import Path
from numpy import array, unique, median, sort, mean, nanstd, nanmean, any, random, nan, zeros
from scipy.stats import spearmanr
from pandas import DataFrame
from torch import nn as nn
from scripts.corpora import Corpus
from sklearn.metrics import silhouette_samples


def save_results(results_dict, save_path, label):
    skip_results = {k.replace('skip_', ''): v for k, v in results_dict.items() if k.startswith('skip_')}
    lstm_results = {k.replace('lstm_', ''): v for k, v in results_dict.items() if k.startswith('lstm_')}
    skip_df, lstm_df = DataFrame(skip_results), DataFrame(lstm_results)
    skip_df.to_csv(save_path / f'skip_{label}.csv', index=False)
    lstm_df.to_csv(save_path / f'lstm_{label}.csv', index=False)
    return skip_df, lstm_df


def silhouette_score(embeddings, words, aggregation='median'):
    words = words[words['category'] != 'otro']
    embeddings = array([embeddings[w] for w in words.index])
    labels = array(words['category'])
    silhouette_vals = silhouette_samples(embeddings, labels, metric='cosine')

    category_scores = {}
    for category in unique(labels):
        mask = labels == category
        vals = silhouette_vals[mask]

        if aggregation == 'median':
            category_scores[category] = median(vals)
        elif aggregation == 'trimmed_mean':
            # Remove top/bottom 10%
            sorted_vals = sort(vals)
            if len(vals) < 10:
                category_scores[category] = mean(sorted_vals)
            else:
                trim = int(0.1 * len(vals))
                category_scores[category] = mean(sorted_vals[trim:-trim])

    return category_scores


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


def in_off_stimuli_word_pairs(words_with_measurements, words_in_stimuli, words_similarities, resamples, seed=42):
    in_stimuli = words_similarities[(words_similarities['word1'].isin(words_with_measurements))
                                    & (words_similarities['word2'].isin(words_with_measurements))]
    off_stimuli = words_similarities[(~words_similarities['word1'].isin(words_in_stimuli))
                                     & (~words_similarities['word2'].isin(words_in_stimuli))]
    rng = random.default_rng(seed)
    seeds = rng.integers(0, 10000, size=resamples)
    in_stimuli_wp, off_stimuli_wp = [], []
    for seed in seeds:
        in_stimuli_wp.append(in_stimuli.sample(frac=1, replace=True, random_state=seed))
        off_stimuli_wp.append(off_stimuli.sample(frac=1, replace=True, random_state=seed))

    return in_stimuli_wp, off_stimuli_wp


def similarities(words_vectors, words, answers):
    similarities = zeros(len(answers))
    for i, word_pair in enumerate(zip(words, answers)):
        word, answer = word_pair
        similarities[i] = word_similarity(words_vectors, word, answer)
    return similarities


def word_similarity(words_vectors, word, answer):
    if answer is None or answer not in words_vectors or word not in words_vectors:
        return nan
    return words_vectors.similarity(word, answer)


def embeddings(words_vectors, words):
    embeddings, corresponding_words = [], []
    for word in words:
        if word in words_vectors:
            embeddings.append(words_vectors[word])
            corresponding_words.append(word)
    return array(embeddings), corresponding_words


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
    if any(fix_corrs):
        for i in range(n_gaze_features):
            print(f'{gaze_features[i]} correlation: {nanmean(fix_corrs[i]):.3f} '
                  f'(+/- {nanstd(fix_corrs[i]):.3f}) | p-value: {nanmean(fix_pvalues[i]):.3f} '
                  f'(+/- {nanstd(fix_pvalues[i]):.3f})')
