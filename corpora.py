from gensim.corpora.wikicorpus import WikiCorpus
from gensim.parsing import preprocessing
from itertools import chain

CHARS_MAP = {'—': '', '‒': '', '−': '', '-': '', '«': '', '»': '',
             '“': '', '”': '', '\'': '', '\"': '', '‘': '', '’': '',
             '(': '', ')': '', ';': '', ',': '', ':': '', '.': '', '…': '',
             '¿': '', '?': '', '¡': '', '!': '', '=': ''}


class Corpora:
    def __init__(self):
        self.corpora = []

    def add_corpus(self, name, path, split):
        self.corpora.append(Corpus(name, path, split))

    def __iter__(self):
        for sentence in chain.from_iterable(corpus.get_texts() for corpus in self.corpora):
            yield list(sentence)


class Corpus:
    def __init__(self, name, path, split):
        self.name = name
        self.split = split
        self.chars_mapping = str.maketrans(CHARS_MAP)
        self.corpus = self.load_corpus(path)

    def load_corpus(self, path):
        corpus = []
        if self.name == 'wikidump':
            corpus = WikiCorpus(path, dictionary={})
        else:
            if path.exists():
                files = [f for f in path.iterdir()]
                for file in files:
                    if file.is_file():
                        with file.open('r') as f:
                            sentences = f.read().split('.')
                            corpus.extend([self.preprocess_str(sentence)
                                                   for sentence in sentences if len(sentence) > 0])
                    elif file.is_dir():
                        corpus += self.load_corpus(file)
        return corpus

    def preprocess_str(self, string):
        string = string.translate(self.chars_mapping)
        words = preprocessing.split_on_space(string.lower())
        return words

    def get_texts(self):
        if self.name == 'wikidump':
            return self.corpus.get_texts()
        else:
            return self.corpus
