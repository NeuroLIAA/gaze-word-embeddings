import argparse
from pathlib import Path
from scripts.test import test
from scripts.train import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Model base name')
    parser.add_argument('-c', '--corpora', type=str, default='all_wikis+scanpaths',
                        help='Texts to be employed for training')
    parser.add_argument('-s', '--sources', type=str, default='remote+local',
                        help='Corpora data sources. If remote, will fetch from huggingface\'s large_spanish_corpus')
    parser.add_argument('-f', '--fraction', type=float, default=1.0,
                        help='Fraction of baseline corpus to employ for training')
    parser.add_argument('-r', '--repeats', type=int, default=1,
                        help='Number of times the local corpus will be iterated over for training')
    parser.add_argument('-sg', '--skip_gram', action='store_true', help='Use skip-gram instead of CBOW')
    parser.add_argument('-ns', '--negative_samples', type=int, default=20,
                        help='Number of negative samples to be used in training')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('-min', '--min_count', type=int, default=5, help='Minimum number of occurrences for a word')
    parser.add_argument('-size', '--size', type=int, default=300, help='Size of the word vectors')
    parser.add_argument('-w', '--window', type=int, default=5, help='Window size')
    parser.add_argument('-th', '--threads', type=int, default=12, help='Number of workers to use')
    parser.add_argument('-min_token', '--min_token', type=int, default=2,
                        help='Word min length, in tokens')
    parser.add_argument('-max_token', '--max_token', type=int, default=20,
                        help='Word max length, in tokens')
    parser.add_argument('-min_length', '--min_length', type=int, default=10,
                        help='Sentence min length, in tokens, for large scale corpora')
    parser.add_argument('-st', '--stimuli', type=str, default='stimuli',
                        help='Path to item files employed in the experiment')
    parser.add_argument('-em', '--embeddings', type=str, default='evaluation/SWOWRP_embeddings.vec',
                        help='Human derived word embeddings to be used as ground truth for evaluation')
    parser.add_argument('-sa', '--subjs_associations', type=str, default='evaluation/subjects_associations.csv',
                        help='Subjects free associations to words file to be employed for evaluation')
    parser.add_argument('-wa', '--words_associations', type=str, default='evaluation/words_associations.csv',
                        help='Words associations file to be employed for evaluation')
    parser.add_argument('-ws', '--words_samples', type=int, default=1000,
                        help='Number of words to be sampled from the words association file for evaluation')
    parser.add_argument('-mf', '--min_freq', type=int, default=15,
                        help='Minimum number of occurrences for an answer in the words association file for evaluation')
    parser.add_argument('-ss', '--sort_sim_by', type=str, default='texts',
                        help='Sort similarity plots by the specified model values')
    parser.add_argument('-t', '--test', action='store_true', help='Perform model evaluation on all its variations')
    parser.add_argument('-se', '--standard_error', action='store_true', help='Plot error bars in similarity plots')
    parser.add_argument('-o', '--output', type=str, default='models', help='Where to save the trained models')
    args = parser.parse_args()
    output, sa_file, wa_file = Path(args.output), Path(args.subjs_associations), Path(args.words_associations)
    source_labels, corpora_labels = args.sources.split('+'), args.corpora.split('+')
    results_path, stimuli_path, gt_embeddings_file = Path('results'), Path(args.stimuli), Path(args.embeddings)
    if len(source_labels) != len(corpora_labels):
        raise ValueError('You must specify from where each corpus will be fetched')
    if args.fraction < 1.0:
        model_path = output / f'{args.model}_{int(args.fraction * 100)}%'
    else:
        model_path = output / args.model
    if args.test:
        test(model_path, wa_file, sa_file, args.min_freq, args.words_samples, stimuli_path, gt_embeddings_file,
             results_path, args.sort_sim_by, args.standard_error)
    else:
        train(corpora_labels, source_labels, args.fraction, args.repeats, args.skip_gram, args.negative_samples,
              args.epochs, args.threads, args.min_token, args.max_token, args.min_length, args.size, args.window,
              args.min_count, model_path)
