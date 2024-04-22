import argparse
import pandas as pd
from numpy import abs
from scipy.stats import spearmanr
from pathlib import Path
from gensim.models import KeyedVectors
from scripts.utils import similarities, get_model_path, get_words_in_corpus, in_off_stimuli_word_pairs
from scripts.plot import plot_distance_to_gt, similarity_distributions, plot_distance_to_gt_across_thresholds


def test(model_path, wa_file, wf_file, num_samples, sim_threshold, gt_threshold, gt_embeddings_file, stimuli_path,
         save_path, error_bars, seed):
    models = [dir_ for dir_ in sorted(model_path.iterdir()) if dir_.is_dir()]
    if len(models) == 0:
        raise ValueError(f'There are no models in {model_path}')
    if not wa_file.exists():
        raise ValueError(f'Evaluation file missing: {wa_file} does not exist')
    if not stimuli_path.exists():
        raise ValueError(f'Stimuli files missing: {stimuli_path} does not exist')
    words_associations, words_frequency = pd.read_csv(wa_file), pd.read_csv(wf_file)
    gt_word_pairs, in_stimuli, off_stimuli = load_and_evaluate_gt(gt_embeddings_file, stimuli_path, words_associations,
                                                                  words_frequency, num_samples, seed)
    models_results = {'word_pairs': {}, 'gt_similarities': {}}
    for model_dir in models:
        model_wv = KeyedVectors.load_word2vec_format(str(model_dir / f'{model_dir.name}.vec'))
        test_model(model_wv, model_dir.name, words_associations, gt_word_pairs, in_stimuli, off_stimuli,
                   models_results)

    model_basename = model_path.name
    save_path = save_path / model_basename
    save_path.mkdir(exist_ok=True, parents=True)
    print_words_pairs_correlations(models_results['word_pairs'])
    models_thresholds = similarity_distributions(models_results['gt_similarities'], save_path)
    plot_distance_to_gt_across_thresholds(models_results['gt_similarities'], models_thresholds, save_path, error_bars)
    plot_distance_to_gt(models_results['gt_similarities'], sim_threshold, gt_threshold, save_path, error_bars)


def load_and_evaluate_gt(gt_embeddings_file, stimuli_path, words_associations, words_frequency, num_samples, seed):
    gt_embeddings = KeyedVectors.load_word2vec_format(str(gt_embeddings_file))
    words_in_stimuli = get_words_in_corpus(stimuli_path)
    gt_word_pairs, in_stimuli, off_stimuli = in_off_stimuli_word_pairs(words_in_stimuli, words_associations,
                                                                       words_frequency, num_samples, seed)
    gt_word_pairs['sim_gt'] = similarities(gt_embeddings, gt_word_pairs['cue'], gt_word_pairs['answer'])

    return gt_word_pairs, in_stimuli, off_stimuli


def test_model(model_wv, model_name, words_associations, gt_word_pairs, in_stimuli, off_stimuli, models_results):
    wa_in_stimuli = words_associations[words_associations['cue'].isin(in_stimuli)].drop_duplicates(subset=['cue'])
    wa_off_stimuli = words_associations[words_associations['cue'].isin(off_stimuli)].drop_duplicates(subset=['cue'])
    wa_in_stimuli['sim'] = abs(similarities(model_wv, wa_in_stimuli['cue'], wa_in_stimuli['answer']))
    wa_off_stimuli['sim'] = abs(similarities(model_wv, wa_off_stimuli['cue'], wa_off_stimuli['answer']))
    models_results['word_pairs'][model_name] = {'in_stimuli':
                                                    spearmanr(wa_in_stimuli['freq'], wa_in_stimuli['sim']),
                                                'off_stimuli':
                                                    spearmanr(wa_off_stimuli['freq'], wa_off_stimuli['sim'])}
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
    print('Spearman correlation to cue-answer frequency')
    for model in models_results:
        in_stimuli_corr = models_results[model]['in_stimuli']
        off_stimuli_corr = models_results[model]['off_stimuli']
        print(f'{model}:\n     In-stimuli: {in_stimuli_corr:.4f} \n     Off-stimuli: {off_stimuli_corr:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Model base name')
    parser.add_argument('-m', '--models', type=str, default='models',
                        help='Path to the trained models')
    parser.add_argument('-f', '--fraction', type=float, default=0.3,
                        help='Fraction of baseline corpus employed for model training')
    parser.add_argument('-ws', '--words_samples', type=int, default=100,
                        help='Number of words to be sampled from the words association file for evaluation')
    parser.add_argument('-t', '--threshold', type=float, default=0.02,
                        help='Threshold for the similarity values to be considered correct')
    parser.add_argument('-gt', '--gt_threshold', type=float, default=-0.02,
                        help='Threshold for the ground truth embeddings similarity values to be considered correct')
    parser.add_argument('-s', '--stimuli', type=str, default='stimuli',
                        help='Path to item files employed in the experiment')
    parser.add_argument('-e', '--embeddings', type=str, default='evaluation/SWOWRP_embeddings.vec',
                        help='Human derived word embeddings to be used as ground truth for evaluation')
    parser.add_argument('-wa', '--words_associations', type=str, default='evaluation/SWOWRP_words_associations.csv',
                        help='Words associations file to be employed for evaluation')
    parser.add_argument('-wf', '--words_frequency', type=str, default='evaluation/words_freq.csv',
                        help='File containing the frequency of each word in the words associations file')
    parser.add_argument('-plt', '--plot', action='store_true', help='Plot similarity to subjects and answers')
    parser.add_argument('-se', '--standard_error', action='store_false', help='Plot error bars in similarity plots')
    parser.add_argument('-seed', '--seed', type=int, default=42, help='Seed for random sampling')
    parser.add_argument('-o', '--output', type=str, default='results', help='Where to save test results')
    args = parser.parse_args()
    wa_file, wf_file = Path(args.words_associations), Path(args.words_frequency)
    output, stimuli_path, gt_embeddings_file = Path(args.output), Path(args.stimuli), Path(args.embeddings)
    model_path = get_model_path(args.models, args.model_name, args.fraction)

    test(model_path, wa_file, wf_file, args.words_samples, args.threshold, args.gt_threshold,
         gt_embeddings_file, stimuli_path, output, args.standard_error, args.seed)
