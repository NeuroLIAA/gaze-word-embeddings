import argparse
import pandas as pd
import numpy as np
from corpora import Corpora
from pathlib import Path
from gensim.models import Word2Vec


def train(corpus, vector_size, window_size, min_count, file_path):
    model = Word2Vec(sentences=corpus, vector_size=vector_size, window=window_size, min_count=min_count, workers=-1)
    model.save(str(file_path))
    return model


def test(model, words_associations):
    words = words_associations.index
    distances = words_associations.apply(lambda answers: distances(model, words, answers))
    return distances


def distances(model, words, answers):
    for i, answer in enumerate(answers):
        answers[i] = distance_to_word(model, words[i], answer)
    return answers


def distance_to_word(model, word, answer):
    if answer is None or answer not in model.wv or word not in model.wv:
        return np.nan
    return model.wv.distance(word, answer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpora', type=str, default='wikidump+texts',
                        help='Texts to be employed for training')
    parser.add_argument('-s', '--source', type=str, default='huggingface',
                        help='Source for large scale data, either remote or local. Remote options: huggingface')
    parser.add_argument('-m', '--model', type=str, default='wikis_texts', help='Model name')
    parser.add_argument('-f', '--fraction', type=float, default=1.0,
                        help='Fraction of baseline corpus to employ for training')
    parser.add_argument('-min', '--min_count', type=int, default=100, help='Minimum number of occurrences for a word')
    parser.add_argument('-size', '--size', type=int, default=300, help='Size of the word vectors')
    parser.add_argument('-w', '--window', type=int, default=5, help='Window size')
    parser.add_argument('-min_token', '--min_token', type=int, default=2,
                        help='Word min length, in tokens')
    parser.add_argument('-max_token', '--max_token', type=int, default=20,
                        help='Word max length, in tokens')
    parser.add_argument('-min_length', '--min_length', type=int, default=10,
                        help='Sentence min length, in tokens, for large scale corpora')
    parser.add_argument('-wa', '--words_association', type=str, default='evaluation/words_associations.pkl',
                        help='Words association file to be employed for evaluation')
    parser.add_argument('-t', '--test', action='store_true', help='Perform model evaluation')
    parser.add_argument('-o', '--output', type=str, default='model', help='Where to save the trained model')
    args = parser.parse_args()
    model_path, wa_file = Path(args.output, args.model), Path(args.words_association)
    if args.test:
        if not model_path.exists():
            raise ValueError('The specified model does not exist')
        if not wa_file.exists():
            raise ValueError('The specified words association file does not exist')
        model = Word2Vec.load(str(model_path))
        words_associations = pd.read_pickle(wa_file)
        distances = test(model, words_associations)
    else:
        corpora = args.corpora.split('+')
        training_corpora = Corpora(args.min_token, args.max_token, args.min_length)
        for corpus in corpora:
            is_large = 'texts' not in corpus
            source = Path(corpus) if not is_large else Path(args.source)
            training_corpora.add_corpus(corpus, source, args.fraction, is_large)
        model_path.parent.mkdir(exist_ok=True)
        train(training_corpora, args.size, args.window, args.min_count, model_path)
