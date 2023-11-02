import argparse
from pathlib import Path
from test import test
from train import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Model name')
    parser.add_argument('-c', '--corpora', type=str, default='all_wikis+texts_et',
                        help='Texts to be employed for training')
    parser.add_argument('-s', '--source', type=str, default='huggingface',
                        help='Source for large scale data, either remote or local. Remote options: huggingface')
    parser.add_argument('-f', '--fraction', type=float, default=1.0,
                        help='Fraction of baseline corpus to employ for training')
    parser.add_argument('-r', '--repeats', type=int, default=1,
                        help='Number of times the local corpus will be iterated over for training')
    parser.add_argument('-min', '--min_count', type=int, default=100, help='Minimum number of occurrences for a word')
    parser.add_argument('-size', '--size', type=int, default=300, help='Size of the word vectors')
    parser.add_argument('-w', '--window', type=int, default=5, help='Window size')
    parser.add_argument('-min_token', '--min_token', type=int, default=2,
                        help='Word min length, in tokens')
    parser.add_argument('-max_token', '--max_token', type=int, default=20,
                        help='Word max length, in tokens')
    parser.add_argument('-min_length', '--min_length', type=int, default=10,
                        help='Sentence min length, in tokens, for large scale corpora')
    parser.add_argument('-wa', '--words_association', type=str, default='evaluation/words_associations.csv',
                        help='Words association file to be employed for evaluation')
    parser.add_argument('-t', '--test', action='store_true', help='Perform model evaluation')
    parser.add_argument('-o', '--output', type=str, default='models',
                        help='Where to save the trained models and evaluation results')
    args = parser.parse_args()
    model_path, wa_file = Path(args.output, args.model), Path(args.words_association)
    if args.fraction < 1.0:
        model_path = model_path / f'{int(args.fraction * 100)}'
    if args.test:
        test(args.model, model_path, wa_file, save_path=Path('results'))
    else:
        train(args.corpora, args.source, args.fraction, args.repeats, args.min_token, args.max_token, args.min_length,
              args.size, args.window, args.min_count, args.model, model_path)
