import argparse
import pandas as pd
from pathlib import Path
from gensim.models import KeyedVectors
from scripts.utils import similarities, get_model_path, get_words_in_corpus, in_off_stimuli_word_pairs
from scripts.plot import (plot_distance_to_gt, plot_freq_to_sim, plot_similarity, scatterplot_gt_similarities,
                          plot_distance_to_gt_across_thresholds)


def test(model_path, wa_file, sa_file, wf_file, min_freq, num_samples, sim_threshold, gt_threshold, gt_embeddings_file,
         stimuli_path, save_path, sort_sim_by, error_bars, plot_sims, seed):
    models = [dir_ for dir_ in sorted(model_path.iterdir()) if dir_.is_dir()]
    if len(models) == 0:
        raise ValueError(f'There are no models in {model_path}')
    if not (sa_file.exists() or wa_file.exists()):
        raise ValueError(f'Evaluation files missing: {wa_file} and {sa_file} do not exist')
    if not stimuli_path.exists():
        raise ValueError(f'Stimuli files missing: {stimuli_path} does not exist')
    subjs_associations = pd.read_csv(sa_file, index_col=0)
    words_associations, words_frequency = pd.read_csv(wa_file), pd.read_csv(wf_file)
    gt_word_pairs = load_and_evaluate_gt(gt_embeddings_file, stimuli_path, words_associations, words_frequency,
                                         num_samples, seed)
    models_results = {'similarity_to_subjs': {}, 'similarity_to_answers': {}, 'word_pairs': {}, 'gt_similarities': {}}
    for model_dir in models:
        model_wv = KeyedVectors.load_word2vec_format(str(model_dir / f'{model_dir.name}.vec'))
        test_model(model_wv, model_dir.name, words_associations, subjs_associations, gt_word_pairs, models_results)

    model_basename = model_path.name
    save_path = save_path / model_basename
    save_path.mkdir(exist_ok=True, parents=True)
    if plot_sims:
        plot_similarity(model_basename, models_results['similarity_to_subjs'], sim_threshold,
                        save_path, sort_sim_by, error_bars)
        plot_freq_to_sim(model_basename, models_results['similarity_to_answers'], save_path, min_appearances=min_freq)
    scatterplot_gt_similarities(models_results['gt_similarities'], save_path)
    gt_thresholds = [-0.25584102, -0.05029088, -0.03782241, -0.03011695, -0.02433026, -0.01961701,
                     -0.01555011, -0.0118527, -0.00845822, -0.005216, -0.0020311, 0.00116537,
                      0.00445931, 0.00795935, 0.01180585, 0.01620096, 0.02155152, 0.02865453,
                      0.03971244, 0.06509454]
    sim_thresholds = [-0.5182156, -0.17888092, -0.13135286, -0.09706547, -0.06829848, -0.04242247,
                      -0.01816167, 0.00529253, 0.02842446, 0.05169781, 0.07547412, 0.10020889,
                      0.12632117, 0.15428203, 0.18484216, 0.218885, 0.25775304, 0.30390432,
                      0.3624946, 0.44852245]
    plot_distance_to_gt_across_thresholds(models_results['gt_similarities'], sim_thresholds, gt_thresholds, save_path, error_bars)
    plot_distance_to_gt(models_results['gt_similarities'], sim_threshold, gt_threshold, save_path, error_bars)
    print_words_pairs_correlations(models_results['word_pairs'])


def load_and_evaluate_gt(gt_embeddings_file, stimuli_path, words_associations, words_frequency, num_samples, seed):
    gt_embeddings = KeyedVectors.load_word2vec_format(str(gt_embeddings_file))
    words_in_stimuli = get_words_in_corpus(stimuli_path)
    gt_word_pairs = in_off_stimuli_word_pairs(words_in_stimuli, words_associations, words_frequency, num_samples, seed)
    gt_word_pairs['sim_gt'] = similarities(gt_embeddings, gt_word_pairs['cue'], gt_word_pairs['answer'])

    return gt_word_pairs


def test_model(model_wv, model_name, words_associations, subjs_associations, gt_word_pairs, models_results):
    answers_sim = similarities(model_wv, words_associations['cue'], words_associations['answer'])
    wa_model_sim = words_associations.copy()
    wa_model_sim['similarity'] = answers_sim
    subjs_cues = subjs_associations.index
    sa_subj_sim = subjs_associations.copy().apply(lambda answers: similarities(model_wv, subjs_cues, answers))
    models_results['similarity_to_subjs'][model_name] = sa_subj_sim
    models_results['similarity_to_answers'][model_name] = wa_model_sim
    models_results['word_pairs'][model_name] = evaluate_word_pairs(model_wv, wa_model_sim)
    models_results['gt_similarities'][model_name] = gt_similarities(gt_word_pairs, model_wv)


def gt_similarities(word_pairs, words_vectors):
    gt_similarities = word_pairs.copy()
    gt_similarities['sim'] = similarities(words_vectors, gt_similarities['cue'], gt_similarities['answer'])

    return gt_similarities


def evaluate_word_pairs(words_vectors, freq_similarity_pairs):
    temp_file = Path('word_pairs.csv')
    word_pairs = freq_similarity_pairs.drop(columns=['similarity', 'n'])
    word_pairs.to_csv(temp_file, index=False, header=False)
    pearson, spearman, oov_ratio = words_vectors.evaluate_word_pairs(temp_file, delimiter=',')
    temp_file.unlink()
    return pearson, spearman, [oov_ratio]


def print_words_pairs_correlations(models_results):
    print('\n---Correlation between similarity and frequency of response for cue-answer pairs---')
    measures = ['Pearson', 'Spearman rank-order', 'Out of vocabulary ratio']
    for i, measure in enumerate(measures):
        print(f'{measure}')
        for model in models_results:
            model_correlations = models_results[model][i]
            print(f'{model}: {model_correlations[0]:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Model base name')
    parser.add_argument('-m', '--models', type=str, default='models',
                        help='Path to the trained models')
    parser.add_argument('-f', '--fraction', type=float, default=0.3,
                        help='Fraction of baseline corpus employed for model training')
    parser.add_argument('-ws', '--words_samples', type=int, default=100,
                        help='Number of words to be sampled from the words association file for evaluation')
    parser.add_argument('-mf', '--min_freq', type=int, default=25,
                        help='Minimum number of occurrences for an answer in the words association file for evaluation')
    parser.add_argument('-t', '--threshold', type=float, default=0.02,
                        help='Threshold for the similarity values to be considered correct')
    parser.add_argument('-gt', '--gt_threshold', type=float, default=-0.02,
                        help='Threshold for the ground truth embeddings similarity values to be considered correct')
    parser.add_argument('-s', '--stimuli', type=str, default='stimuli',
                        help='Path to item files employed in the experiment')
    parser.add_argument('-e', '--embeddings', type=str, default='evaluation/SWOWRP_embeddings.vec',
                        help='Human derived word embeddings to be used as ground truth for evaluation')
    parser.add_argument('-sa', '--subjs_associations', type=str, default='evaluation/subjects_associations.csv',
                        help='Subjects free associations to words file to be employed for evaluation')
    parser.add_argument('-wa', '--words_associations', type=str, default='evaluation/SWOWRP_words_associations.csv',
                        help='Words associations file to be employed for evaluation')
    parser.add_argument('-wf', '--words_frequency', type=str, default='evaluation/words_freq.csv',
                        help='File containing the frequency of each word in the words associations file')
    parser.add_argument('-plt', '--plot', action='store_true', help='Plot similarity to subjects and answers')
    parser.add_argument('-ss', '--sort_sim_by', type=str, default='texts',
                        help='Sort similarity plots by the specified model values')
    parser.add_argument('-se', '--standard_error', action='store_false', help='Plot error bars in similarity plots')
    parser.add_argument('-seed', '--seed', type=int, default=42, help='Seed for random sampling')
    parser.add_argument('-o', '--output', type=str, default='results', help='Where to save test results')
    args = parser.parse_args()
    sa_file, wa_file, wf_file = Path(args.subjs_associations), Path(args.words_associations), Path(args.words_frequency)
    output, stimuli_path, gt_embeddings_file = Path(args.output), Path(args.stimuli), Path(args.embeddings)
    model_path = get_model_path(args.models, args.model_name, args.fraction)

    test(model_path, wa_file, sa_file, wf_file, args.min_freq, args.words_samples, args.threshold, args.gt_threshold,
         gt_embeddings_file, stimuli_path, output, args.sort_sim_by, args.standard_error, args.plot, args.seed)
