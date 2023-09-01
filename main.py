import argparse
from pathlib import Path
from gensim.models import Word2Vec
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.parsing import preprocessing

CHARS_MAP = {'—': '', '‒': '', '−': '', '-': '', '«': '', '»': '',
             '“': '', '”': '', '\'': '', '\"': '', '‘': '', '’': '',
             '(': '', ')': '', ';': '', ',': '', ':': '', '.': '', '…': '',
             '¿': '', '?': '', '¡': '', '!': '', '=': ''}


class WikiIterator:
    def __init__(self, wikicorpus):
        self.corpus = WikiCorpus(wikicorpus, metadata=False)

    def __iter__(self):
        for sentence in self.corpus.get_texts():
            yield list(sentence)


def train(corpus, file_path):
    model = Word2Vec(sentences=corpus, vector_size=300, window=5, min_count=1, workers=-1)
    model.save(str(file_path))
    return model


def load_corpus(path, chars_mapping, split):
    corpus = {'text': []}
    if path.exists():
        files = [f for f in path.iterdir()]
        for file in files:
            if file.is_file():
                with file.open('r') as f:
                    sentences = f.read().split('.')
                    corpus['text'].extend([preprocess_str(sentence, chars_mapping)
                                           for sentence in sentences if len(sentence) > 0])
            elif file.is_dir():
                corpus['text'] += load_corpus(file, chars_mapping, split)['text']
    return corpus


def preprocess_str(string, chars_mapping):
    string = string.translate(chars_mapping)
    words = preprocessing.split_on_space(string.lower())
    return {'text': words}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpora', type=str, default='wikidump+texts',
                        help='Texts to be employed for training')
    parser.add_argument('-d', '--dataset', type=str, default='large_datasets/eswiki-20230820-pages-articles.xml.bz2',
                        help='Path to large text dataset')
    parser.add_argument('-m', '--model', type=str, default='wikis_texts', help='Model name')
    parser.add_argument('-s', '--split', type=float, default=0.1, help='Split for baseline corpus')
    parser.add_argument('-o', '--output', type=str, default='model', help='Where to save the trained model')
    args = parser.parse_args()
    model_path = Path(args.output, args.model)

    chars_mapping = str.maketrans(CHARS_MAP)
    corpora = args.corpora.split('+')
    training_corpus = WikiCorpus(args.dataset)
    model_path.parent.mkdir(exist_ok=True)
    train(training_corpus, model_path)
