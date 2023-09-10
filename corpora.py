from gensim.corpora.wikicorpus import WikiCorpus
from gensim.utils import simple_preprocess
from itertools import chain, islice
from datasets import load_dataset
import regex as re

N_WIKI_ARTICLES = 1889000


class Corpora:
    def __init__(self):
        self.corpora = []

    def add_corpus(self, name, path, split):
        self.corpora.append(Corpus(name, path, split))

    def __iter__(self):
        for sentence in chain.from_iterable(corpus.get_texts() for corpus in self.corpora):
            yield list(sentence['text'])


class Corpus:
    def __init__(self, name, path, split):
        self.name = name
        self.split = split
        self.corpus = self.load_corpus(path)

    def load_corpus(self, path):
        corpus = {'text': []}
        if self.name == 'wikidump':
            corpus = WikiCorpus(path, dictionary={}, article_min_tokens=100)
        elif self.name == 'all_wikis':
            corpus = load_hg_dataset(self.name)
        else:
            if path.exists():
                files = [f for f in path.iterdir()]
                for file in files:
                    if file.is_file():
                        with file.open('r') as f:
                            sentences = f.read().split('.')
                            corpus['text'].extend([preprocess_str({'text': sentence})['text']
                                                   for sentence in sentences if len(sentence) > 0])
                    elif file.is_dir():
                        corpus['text'] += self.load_corpus(file)['text']
        return corpus

    def get_texts(self):
        if self.name == 'wikidump':
            if 0 < self.split < 1.0:
                return islice(self.corpus.get_texts(), int(N_WIKI_ARTICLES * self.split))
            else:
                return self.corpus.get_texts()
        else:
            return self.corpus


def load_hg_dataset(name):
    corpus = load_dataset('large_spanish_corpus', name=name, split='train', streaming=True)
    corpus = corpus.map(preprocess_str)
    corpus = corpus.filter(lambda row: len(row['text']) > 10)
    return corpus


def preprocess_str(string):
    string['text'] = re.sub(r'[^ \nA-Za-zÀ-ÖØ-öø-ÿ/]+', '', string['text'])
    string['text'] = simple_preprocess(string['text'], min_len=2, max_len=20)
    return string