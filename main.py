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
    model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=-1)
    model.save(str(file_path))
    return model


def load_baseline_corpus(dataset='large_spanish_corpus', name='all_wikis', split='10%'):
    baseline_corpus = load_dataset(dataset, name, split=f'train[:{split}]')['text']
    baseline_corpus = [preprocess_str(text, chars_mapping) for text in baseline_corpus if len(text) > 0]
    return baseline_corpus


def load_corpus(name, chars_mapping, split='10%'):
    if not Path(name).exists():
        corpus = load_baseline_corpus(name=name, split=split)
    else:
        files = [f for f in Path(name).iterdir()]
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
    parser.add_argument('-c', '--corpora', type=str, default='all_wikis+texts',
                        help='  Texts to be employed for training')
    parser.add_argument('-m', '--model', type=str, default='wikis_texts', help='Model name')
    parser.add_argument('-s', '--split', type=str, default='10%', help='Split for baseline corpus')
    parser.add_argument('-o', '--output', type=str, default='model', help='Where to save the trained model')
    args = parser.parse_args()
    model_path = Path(args.output, args.model)

    chars_mapping = str.maketrans(CHARS_MAP)
    corpora = args.corpora.split('+')
    training_corpus = []
    for corpus_name in corpora:
        training_corpus += load_corpus(corpus_name, chars_mapping, args.split)
    train(training_corpus, model_path)
