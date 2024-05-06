import argparse
import pandas as pd
from numpy import abs, random
from scipy.stats import spearmanr
from pathlib import Path
from gensim.models import KeyedVectors
from scripts.utils import similarities, get_model_path, get_words_in_corpus, in_off_stimuli_word_pairs
from scripts.plot import plot_correlations
from scripts.process_swow import load_swow


def test(model_path, words_associations, words_freq, num_samples, stimuli_path, save_path, seed):
    models = [dir_ for dir_ in sorted(model_path.iterdir()) if dir_.is_dir()]
    if len(models) == 0:
        raise ValueError(f'There are no models in {model_path}')
    if not stimuli_path.exists():
        raise ValueError(f'Stimuli files missing: {stimuli_path} does not exist')
    rng = random.default_rng(seed)
    in_stimuli_words, off_stimuli_words = load_and_evaluate_gt(stimuli_path, words_associations,
                                                                  words_freq, num_samples, rng)
    models_results = {'in_stimuli': {}, 'off_stimuli': {}}
    for model_dir in models:
        model_wv = KeyedVectors.load_word2vec_format(str(model_dir / f'{model_dir.name}.vec'))
        test_model(model_wv, model_dir.name, in_stimuli_words, off_stimuli_words, models_results)

    model_basename = model_path.name
    save_path = save_path / model_basename
    save_path.mkdir(exist_ok=True, parents=True)
    plot_correlations(models_results, save_path)


def load_and_evaluate_gt(stimuli_path, words_associations, words_frequency, num_samples, rng):
    words_in_stimuli = get_words_in_corpus(stimuli_path)
    in_stimuli_wp, off_stimuli_wp = in_off_stimuli_word_pairs(words_in_stimuli, words_associations, words_frequency,
                                                              num_samples, rng)

    return in_stimuli_wp, off_stimuli_wp


def test_model(model_wv, model_name, in_stimuli_wp, off_stimuli_wp, models_results):
    models_results['in_stimuli'][model_name] = []
    models_results['off_stimuli'][model_name] = []
    for in_stimuli, off_stimuli in zip(in_stimuli_wp, off_stimuli_wp):
        in_stimuli['sim'] = abs(similarities(model_wv, in_stimuli['cue'], in_stimuli['answer']))
        off_stimuli['sim'] = abs(similarities(model_wv, off_stimuli['cue'], off_stimuli['answer']))
        models_results['in_stimuli'][model_name].append(spearmanr(in_stimuli['freq'], in_stimuli['sim'],
                                                                             nan_policy='omit').statistic)
        models_results['off_stimuli'][model_name].append(spearmanr(off_stimuli['freq'], off_stimuli['sim'],
                                                                                nan_policy='omit').statistic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Model base name')
    parser.add_argument('-m', '--models', type=str, default='models', help='Path to trained models')
    parser.add_argument('-f', '--fraction', type=float, default=0.3,
                        help='Fraction of baseline corpus employed for model training')
    parser.add_argument('-ws', '--words_samples', type=int, default=100,
                        help='Number of words to be sampled from the words association file for evaluation')
    parser.add_argument('-s', '--stimuli', type=str, default='stimuli',
                        help='Path to item files employed in the experiment')
    parser.add_argument('-wa', '--words_associations', type=str, default='evaluation/SWOWRP_words_associations.csv',
                        help='Words associations file to be employed for evaluation')
    parser.add_argument('-wf', '--words_frequency', type=str, default='evaluation/words_freq.csv',
                        help='File containing the frequency of each word in the words associations file')
    parser.add_argument('-min_freq', '--min_freq', type=int, default=2,
                        help='Minimum frequency of answer for a cue answer pair to be considered')
    parser.add_argument('-set', '--set', type=str, default='val', help='Set to evaluate')
    parser.add_argument('-seed', '--seed', type=int, default=42, help='Seed for random sampling')
    parser.add_argument('-o', '--output', type=str, default='results', help='Where to save test results')
    args = parser.parse_args()
    words_freq = pd.read_csv(args.words_frequency)
    output, stimuli_path = Path(args.output), Path(args.stimuli)
    swow = load_swow(args.words_associations, words_freq, stimuli_path, args.set, args.min_freq, args.seed)
    model_path = get_model_path(args.models, args.model_name, args.fraction)

    test(model_path, swow, words_freq, args.words_samples, stimuli_path, output, args.seed)
