import argparse
import pandas as pd
from numpy import abs, random, std
from scipy.stats import spearmanr
from pathlib import Path
from gensim.models import KeyedVectors
from scripts.utils import similarities, get_embeddings_path, get_words_in_corpus, in_off_stimuli_word_pairs, embeddings
from scripts.CKA import linear_CKA
from scripts.plot import plot_correlations
from scripts.process_swow import load_swow


def test(embeddings_path, words_associations, swow_wv, words_freq, num_samples, resamples, stimuli_path, gaze_table,
         save_path, seed):
    models = [dir_ for dir_ in sorted(embeddings_path.iterdir()) if dir_.is_dir()]
    if len(models) == 0:
        raise ValueError(f'There are no models in {embeddings_path}')
    if not stimuli_path.exists():
        raise ValueError(f'Stimuli files missing: {stimuli_path} does not exist')
    rng = random.default_rng(seed)
    words_in_stimuli = get_words_in_corpus(stimuli_path)
    words_with_measurements = [word for word in gaze_table.index if word in words_in_stimuli]
    embeddings_in_stimuli, corresponding_words = embeddings(swow_wv, words_with_measurements)
    in_stimuli_wp, off_stimuli_wp = in_off_stimuli_word_pairs(words_with_measurements, words_associations, words_freq,
                                                              num_samples, resamples, rng)
    models_results = {'in_stimuli': {}, 'off_stimuli': {}}
    for model_dir in models:
        model_wv = KeyedVectors.load_word2vec_format(str(model_dir / f'{model_dir.name}.vec'))
        test_word_pairs(model_wv, model_dir.name, in_stimuli_wp, off_stimuli_wp, models_results)
        model_embeddings = model_wv[corresponding_words]
        mean_cka, std_cka = compare_distributions(model_embeddings, embeddings_in_stimuli, num_samples, resamples, seed)
        print(f'{model_dir.name} CKA with SWOW embeddings: {mean_cka:.4f} (+/- {std_cka:.4f})')

    model_basename = embeddings_path.name
    save_path = save_path / model_basename
    save_path.mkdir(exist_ok=True, parents=True)
    plot_correlations(models_results, save_path)


def compare_distributions(model_embeddings, embeddings_in_stimuli, num_samples, resamples, seed):
    rng = random.default_rng(seed)
    linear_ckas = []
    for _ in range(resamples):
        sample = rng.choice(len(model_embeddings), min(num_samples, len(model_embeddings)),
                            replace=False)
        linear_ckas.append(linear_CKA(model_embeddings[sample], embeddings_in_stimuli[sample]))
    return sum(linear_ckas) / len(linear_ckas), std(linear_ckas)


def test_word_pairs(model_wv, model_name, in_stimuli_wp, off_stimuli_wp, models_results):
    models_results['in_stimuli'][model_name] = []
    models_results['off_stimuli'][model_name] = []
    for in_stimuli, off_stimuli in zip(in_stimuli_wp, off_stimuli_wp):
        in_stimuli_sim = similarities(model_wv, in_stimuli['cue'], in_stimuli['answer'])
        off_stimuli_sim = similarities(model_wv, off_stimuli['cue'], off_stimuli['answer'])
        models_results['in_stimuli'][model_name].append(spearmanr(in_stimuli['freq'], in_stimuli_sim,
                                                                  nan_policy='omit').statistic)
        models_results['off_stimuli'][model_name].append(spearmanr(off_stimuli['freq'], off_stimuli_sim,
                                                                   nan_policy='omit').statistic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='Training data descriptive name')
    parser.add_argument('-e', '--embeddings', type=str, default='embeddings', help='Path to extracted embeddings')
    parser.add_argument('-f', '--fraction', type=float, default=1.0,
                        help='Fraction of baseline corpus employed for model training')
    parser.add_argument('-ws', '--words_samples', type=int, default=500,
                        help='Number of words to be sampled from the words association file for evaluation')
    parser.add_argument('-rs', '--resample', type=int, default=100, help='Number of times to resample word pairs')
    parser.add_argument('-s', '--stimuli', type=str, default='stimuli',
                        help='Path to item files employed in the experiment')
    parser.add_argument('-t', '--gaze_table', type=str, default='words_measurements.pkl',
                        help='Path to file with words gaze measurements')
    parser.add_argument('-wa', '--words_associations', type=str, default='evaluation/SWOWRP_words_associations.csv',
                        help='Words associations file to be employed for evaluation')
    parser.add_argument('-gt', '--ground_truth', type=str, default='evaluation/SWOWRP_embeddings.vec',
                        help='Ground truth embeddings for evaluation')
    parser.add_argument('-min_freq', '--min_freq', type=float, default=0.02,
                        help='Minimum frequency of answer for a cue answer pair to be considered')
    parser.add_argument('-wf', '--words_frequency', type=str, default='evaluation/words_freq.csv',
                        help='File containing the frequency of each cue in the words associations file')
    parser.add_argument('-nc', '--non_content', type=str, default='evaluation/non_content_cues.csv',
                        help='File containing a list of non-content cues to be filtered out')
    parser.add_argument('-set', '--set', type=str, default='val', help='Set to evaluate')
    parser.add_argument('-seed', '--seed', type=int, default=42, help='Seed for random sampling')
    parser.add_argument('-o', '--output', type=str, default='results', help='Where to save test results')
    args = parser.parse_args()
    words_freq = pd.read_csv(args.words_frequency)
    output, stimuli_path = Path(args.output), Path(args.stimuli)
    swow = load_swow(args.words_associations, words_freq, args.non_content, args.min_freq, stimuli_path,
                     args.set, args.seed)
    swow_wv = KeyedVectors.load_word2vec_format(args.ground_truth)
    embeddings_path = get_embeddings_path(args.embeddings, args.data, args.fraction)
    gaze_table = pd.read_pickle(args.gaze_table)

    test(embeddings_path, swow, swow_wv, words_freq, args.words_samples, args.resample, stimuli_path, gaze_table,
         output, args.seed)
