import argparse
import pandas as pd
import numpy as np
from corpora import Corpora
from eval import answers_frequency
from pathlib import Path
from gensim.models import Word2Vec


def train(corpora, source, fraction, min_token_len, max_token_len, min_sentence_len,
          vector_size, window_size, min_count, save_path):
    corpora = corpora.split('+')
    corpora = load_corpora(corpora, source, fraction, min_token_len, max_token_len, min_sentence_len)
    model = Word2Vec(sentences=corpora, vector_size=vector_size, window=window_size, min_count=min_count, workers=-1)
    save_path.mkdir(exist_ok=True, parents=True)
    model.save(str(save_path / f'{save_path.name}.model'))
    return model


def load_corpora(corpora, source, fraction, min_token_len, max_token_len, min_sentence_len):
    training_corpora = Corpora(min_token_len, max_token_len, min_sentence_len)
    for corpus in corpora:
        is_large = 'texts' not in corpus
        source = Path(corpus) if not is_large else Path(source)
        training_corpora.add_corpus(corpus, source, fraction, is_large)
    return training_corpora


def test(model_path, wa_file):
    model_file = model_path / f'{model_path.name}.model'
    if not model_file.exists():
        raise ValueError('The specified models does not exist')
    if not wa_file.exists():
        raise ValueError('The specified words association file does not exist')
    model = Word2Vec.load(str(model_file))
    words_associations = pd.read_csv(wa_file, index_col=0)
    words = words_associations.index
    similarities_df = words_associations.apply(lambda answers: similarities(model, words, answers))
    frequency = answers_frequency(words_associations)
    save_path = model_path / 'test'
    save_path.mkdir(exist_ok=True)
    similarities_df.to_pickle(save_path / f'{wa_file.stem}.pkl')
    return similarities_df


def similarities(model, words, answers):
    for i, answer in enumerate(answers):
        answers[i] = word_similarity(model, words[i], answer)
    return answers


def word_similarity(model, word, answer):
    if answer is None or answer not in model.wv or word not in model.wv:
        return np.nan
    return model.wv.similarity(word, answer)


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
    parser.add_argument('-t', '--test', action='store_true', help='Perform models evaluation')
    parser.add_argument('-o', '--output', type=str, default='models', help='Where to save the trained models')
    args = parser.parse_args()
    model_path, wa_file = Path(args.output, args.model), Path(args.words_association)
    if args.test:
        test(model_path, wa_file)
    else:
        train(args.corpora, args.source, args.fraction, args.min_token, args.max_token, args.min_length,
              args.size, args.window, args.min_count, model_path)
