import argparse
from pathlib import Path
from gensim.models import Word2Vec
from gensim.parsing import preprocessing
from datasets import load_dataset

CHARS_MAP = {'—': '', '‒': '', '−': '', '-': '', '«': '', '»': '',
             '“': '', '”': '', '\'': '', '\"': '', '‘': '', '’': '',
             '(': '', ')': '', ';': '', ',': '', ':': '', '.': '', '…': '',
             '¿': '', '?': '', '¡': '', '!': '', '=': ''}


def train(corpus, file_path):
    model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=5, workers=-1)
    model.save(str(file_path))
    return model


def load_baseline_corpus(dataset='large_spanish_corpus', name='all_wikis', split='10%'):
    baseline_corpus = load_dataset(dataset, name, split=f'train[:{split}]')['text']
    baseline_corpus = [preprocess_str(text, chars_mapping) for text in baseline_corpus if len(text) > 0]
    return baseline_corpus


def load_corpus(path, chars_mapping):
    files = [f for f in path.iterdir()]
    corpus = []
    for file in files:
        if file.is_file():
            with file.open('r') as f:
                sentences = f.read().split('.')
                corpus.extend([preprocess_str(sentence, chars_mapping) for sentence in sentences if len(sentence) > 0])
        elif file.is_dir():
            corpus += load_corpus(file, chars_mapping)
    return corpus


def preprocess_str(string, chars_mapping):
    string = string.translate(chars_mapping)
    return preprocessing.split_on_space(string.lower())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', type=str, default='corpus', help='Path to training corpus')
    parser.add_argument('-m', '--model', type=str, default='w2v', help='Model name')
    parser.add_argument('-o', '--output', type=str, default='model', help='Where to save the trained model')
    parser.add_argument('-d', '--dataset', type=str, default='all_wikis',
                        help='Dataset name of baseline corpus')
    parser.add_argument('-s', '--split', type=str, default='10%', help='Split for baseline corpus')
    args = parser.parse_args()

    corpus_path, model_path, baseline_path = (Path(args.corpus), Path(args.output, args.model),
                                              Path(args.output, 'baseline'))
    chars_mapping = str.maketrans(CHARS_MAP)
    baseline_corpus = load_baseline_corpus(name=args.dataset, split=args.split)
    corpus = load_corpus(corpus_path, chars_mapping)
    train(baseline_corpus + corpus, model_path)
