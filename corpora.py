from gensim.utils import simple_preprocess
from itertools import chain, islice
from datasets import load_dataset
import regex as re


class Corpora:
    def __init__(self):
        self.corpora = []

    def add_corpus(self, name, source, fraction, is_large):
        self.corpora.append(Corpus(name, source, fraction, is_large))

    def __iter__(self):
        for sentence in chain.from_iterable(corpus.get_texts() for corpus in self.corpora):
            yield list(sentence['text'])


class Corpus:
    def __init__(self, name, source, fraction, is_large):
        self.name = name
        self.fraction = fraction
        self.is_large = is_large
        self.corpus = self.load_corpus(source)

    def load_corpus(self, source):
        corpus = {'text': []}
        if source.name == 'huggingface':
            corpus = load_hg_dataset(self.name)
        else:
            self.load_local_corpus(source, corpus)
        return corpus

    def get_texts(self):
        if self.is_large and 0 < self.fraction < 1.0:
            return islice(self.corpus, int(self.corpus.info.splits['train'].num_examples * self.fraction))
        else:
            return self.corpus

    def load_local_corpus(self, source, corpus):
        if source.exists():
            files = [f for f in source.iterdir()]
            for file in files:
                if file.is_file():
                    with file.open('r') as f:
                        sentences = f.read().split('.')
                        corpus['text'].extend([preprocess_str({'text': sentence})['text']
                                               for sentence in sentences if len(sentence) > 0])
                elif file.is_dir():
                    corpus['text'] += self.load_corpus(file)['text']


def load_hg_dataset(name):
    corpus = load_dataset('large_spanish_corpus', name=name, split='train', streaming=True)
    corpus = corpus.map(preprocess_str)
    corpus = corpus.filter(lambda row: len(row['text']) > 10)
    return corpus


def preprocess_str(string):
    string['text'] = re.sub(r'[^ \nA-Za-zÀ-ÖØ-öø-ÿ/]+', '', string['text'])
    string['text'] = simple_preprocess(string['text'], min_len=2, max_len=20)
    return string
