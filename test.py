import argparse
from pandas import read_csv
from numpy import random
from scipy.stats import spearmanr
from pathlib import Path
from scripts.keyedvectors import KeyedVectors
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scripts.utils import (similarities, get_embeddings_path, get_words_in_corpus, in_off_stimuli_word_pairs,
                           embeddings, silhouette_score)
from scripts.CKA import linear_CKA
from scripts.plot import plot_distribution, plot_embeddings, plot_semantic_scores


def test(embeddings_path, words_similarities_file, swow_wv, resamples, stimuli_path, gaze_table,
         non_content_words, save_path, seed, silent):
    models = [dir_ for dir_ in sorted(embeddings_path.iterdir()) if dir_.is_dir()]
    if len(models) == 0:
        raise ValueError(f'There are no models in {embeddings_path}')
    if not stimuli_path.exists():
        raise ValueError(f'Stimuli files missing: {stimuli_path} does not exist')
    words_in_stimuli = get_words_in_corpus(stimuli_path)
    words_similarities = read_csv(words_similarities_file)
    words_with_measures = [word for word in gaze_table.index if word in words_in_stimuli]
    in_stimuli_wp, off_stimuli_wp = in_off_stimuli_word_pairs(words_with_measures, words_in_stimuli,
                                                              words_similarities, resamples, seed)
    content_words = [word for word in words_with_measures if word not in non_content_words.values]
    gaze_table = gaze_table.loc[content_words]
    gt_embeddings_in_stimuli, corresponding_words = embeddings(swow_wv, content_words)
    save_path = save_path / embeddings_path.name
    save_path.mkdir(exist_ok=True, parents=True)
    semantic_categories = [category for category in gaze_table['category'].unique() if category != 'otro']
    models_results = {'in_stimuli': {}, 'off_stimuli': {}, 'CKA': {}, 'kNN_overlap': {},
                      'semantic_clustering': {category: {} for category in semantic_categories}}
    for model_dir in tqdm(models, desc='Evaluating models'):
        model_wv = KeyedVectors.load_word2vec_format(str(next(model_dir.glob('*.vec'))))
        test_word_pairs(model_wv, model_dir.name, in_stimuli_wp, off_stimuli_wp, models_results)
        model_embeddings = model_wv[corresponding_words]
        linear_ckas = compare_distributions(model_embeddings, gt_embeddings_in_stimuli, resamples, seed)
        models_results['CKA'][model_dir.name] = linear_ckas
        local_overlaps = compare_nearest_neighbors(corresponding_words, model_wv, swow_wv, k=10)
        models_results['kNN_overlap'][model_dir.name] = local_overlaps
        semantic_clustering(model_wv, gaze_table, model_dir.name, models_results['semantic_clustering'], save_path)
        tqdm.write(f'{model_dir.name} done')

    plot_distribution(models_results['CKA'], save_path, label='CKA', ylabel='CKA',
                      fig_title='CKA to SWOW-RP embeddings', silent=silent)
    plot_distribution(models_results['kNN_overlap'], save_path, label='kNN_overlap', ylabel='kNN % Overlap',
                      fig_title='kNN Overlap with SWOW-RP embeddings', silent=silent)
    plot_semantic_scores(models_results['semantic_clustering'], models, save_path, silent=silent)
    save_path = save_path / words_similarities_file.stem
    save_path.mkdir(exist_ok=True, parents=True)
    plot_distribution(models_results['in_stimuli'], save_path, label='word_pairs_in_stimuli', ylabel='Spearman r',
                      fig_title='Word pairs fine-tuned', silent=silent)
    plot_distribution(models_results['off_stimuli'], save_path, label='word_pairs_off_stimuli', ylabel='Spearman r',
                      fig_title='Word pairs off stimuli', silent=silent)


def compare_distributions(model_embeddings, gt_embeddings_in_stimuli, resamples, seed):
    rng = random.default_rng(seed)
    ckas = []
    for _ in tqdm(range(resamples), desc='Resampling'):
        sample = rng.choice(len(model_embeddings), len(model_embeddings), replace=True)
        ckas.append(linear_CKA(model_embeddings[sample], gt_embeddings_in_stimuli[sample]))
    return ckas


def compare_nearest_neighbors(words, model_embeddings, gt_embeddings_in_stimuli, k=10):
    model_nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(model_embeddings[words])
    swow_nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(gt_embeddings_in_stimuli[words])
    _, model_indices = model_nn.kneighbors(model_embeddings[words])
    _, swow_indices = swow_nn.kneighbors(gt_embeddings_in_stimuli[words])
    # Exclude the first nearest neighbor (the word itself)
    model_indices = model_indices[:, 1:]
    swow_indices = swow_indices[:, 1:]
    overlaps = [len(set(model_indices[i]).intersection(set(swow_indices[i]))) / k
                for i in range(len(words))]
    return overlaps


def semantic_clustering(model_wv, words_data, model_name, results_dict, save_path):
    score_per_category = silhouette_score(model_wv, words_data)
    for category, score in score_per_category.items():
        results_dict[category][model_name] = score
    plot_embeddings(model_wv, words_data, model_name, categories_scores=score_per_category, save_path=save_path)


def test_word_pairs(model_wv, model_name, in_stimuli_wp, off_stimuli_wp, models_results):
    models_results['in_stimuli'][model_name] = []
    models_results['off_stimuli'][model_name] = []
    for in_stimuli, off_stimuli in zip(in_stimuli_wp, off_stimuli_wp):
        in_stimuli_sim = similarities(model_wv, in_stimuli['word1'], in_stimuli['word2'])
        off_stimuli_sim = similarities(model_wv, off_stimuli['word1'], off_stimuli['word2'])
        models_results['in_stimuli'][model_name].append(spearmanr(in_stimuli['score'], in_stimuli_sim,
                                                                  nan_policy='omit').statistic)
        models_results['off_stimuli'][model_name].append(spearmanr(off_stimuli['score'], off_stimuli_sim,
                                                                   nan_policy='omit').statistic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='Training data descriptive name')
    parser.add_argument('-e', '--embeddings', type=str, default='embeddings', help='Path to extracted embeddings')
    parser.add_argument('-f', '--fraction', type=float, default=1.0,
                        help='Fraction of baseline corpus employed for model training')
    parser.add_argument('-r', '--resample', type=int, default=100, help='Number of times to resample words')
    parser.add_argument('-s', '--stimuli', type=str, default='stimuli',
                        help='Path to item files employed in the experiment')
    parser.add_argument('-t', '--gaze_table', type=str, default='words_measures.csv',
                        help='Path to file with words gaze measurements')
    parser.add_argument('-ws', '--words_similarities', type=str, default='evaluation/simlex.csv',
                        help='Word pairs similarities file to be employed for evaluation')
    parser.add_argument('-gt', '--ground_truth', type=str, default='evaluation/SWOWRP_embeddings.vec',
                        help='Ground truth embeddings for evaluation')
    parser.add_argument('-nc', '--non_content', type=str, default='evaluation/non_content_words.csv',
                        help='File containing a list of non-content words to be filtered out')
    parser.add_argument('-seed', '--seed', type=int, default=42, help='Seed for random sampling')
    parser.add_argument('-o', '--output', type=str, default='results', help='Where to save test results')
    parser.add_argument('--silent', action='store_true', help='If set, does not show plots')
    args = parser.parse_args()
    output, stimuli_path = Path(args.output), Path(args.stimuli)
    non_content_words = read_csv(args.non_content)['word']
    swow_wv = KeyedVectors.load_word2vec_format(args.ground_truth)
    embeddings_path = get_embeddings_path(args.embeddings, args.data, args.fraction)
    gaze_table = read_csv(args.gaze_table, index_col=0)

    test(embeddings_path, Path(args.words_similarities), swow_wv, args.resample, stimuli_path,
         gaze_table, non_content_words, output, args.seed, args.silent)
