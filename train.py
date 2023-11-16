import logging
import argparse
from scripts.corpora import Corpora
from gensim.models import Word2Vec
from pathlib import Path


def train(corpora_labels, data_sources, fraction, repeats, cbow, negative_samples, epochs, threads,
          min_token_len, max_token_len, min_sentence_len, vector_size, window_size, min_count, save_path):
    print(f'Beginning training with corpora {corpora_labels} ({int(fraction * 100)}% of baseline corpus)')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    corpora = load_corpora(corpora_labels, data_sources, fraction, repeats,
                           min_token_len, max_token_len, min_sentence_len)
    model = Word2Vec(sentences=corpora, sg=not cbow, vector_size=vector_size, window=window_size, min_count=min_count,
                     negative=negative_samples, epochs=epochs, workers=threads)
    model_name, save_path = get_path(save_path, corpora_labels, data_sources)
    save_path.mkdir(exist_ok=True, parents=True)
    model.save(str(save_path / f'{model_name}.model'))
    print(f'Training completed. Model saved at {save_path}')
    return model


def load_corpora(corpora_labels, data_sources, fraction, repeats, min_token_len, max_token_len, min_sentence_len):
    training_corpora = Corpora(min_token_len, max_token_len, min_sentence_len)
    for corpus, source in zip(corpora_labels, data_sources):
        training_corpora.add_corpus(corpus, source, fraction, repeats)
    return training_corpora


def get_path(save_path, corpora_labels, data_sources):
    model_name = corpora_labels[-1] if 'local' in data_sources else 'baseline'
    save_path = save_path / model_name
    return model_name, save_path


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
    parser.add_argument('-cb', '--cbow', action='store_true', help='Use CBOW instead of skip gram')
    parser.add_argument('-ns', '--negative_samples', type=int, default=20,
                        help='Number of negative samples to be used in training')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('-min', '--min_count', type=int, default=20, help='Minimum number of occurrences for a word')
    parser.add_argument('-size', '--size', type=int, default=300, help='Size of the word vectors')
    parser.add_argument('-w', '--window', type=int, default=5, help='Window size')
    parser.add_argument('-t', '--threads', type=int, default=12, help='Number of workers to use')
    parser.add_argument('-min_token', '--min_token', type=int, default=2,
                        help='Word min length, in tokens')
    parser.add_argument('-max_token', '--max_token', type=int, default=20,
                        help='Word max length, in tokens')
    parser.add_argument('-min_length', '--min_length', type=int, default=10,
                        help='Sentence min length, in tokens, for large scale corpora')
    parser.add_argument('-o', '--output', type=str, default='models', help='Where to save the trained models')
    args = parser.parse_args()
    output = Path(args.output)
    source_labels, corpora_labels = args.sources.split('+'), args.corpora.split('+')
    if len(source_labels) != len(corpora_labels):
        raise ValueError('You must specify from where each corpus will be fetched')
    if args.fraction < 1.0:
        model_path = output / f'{args.model}_{int(args.fraction * 100)}%'
    else:
        model_path = output / args.model

    train(corpora_labels, source_labels, args.fraction, args.repeats, args.cbow, args.negative_samples,
          args.epochs, args.threads, args.min_token, args.max_token, args.min_length, args.size, args.window,
          args.min_count, model_path)
