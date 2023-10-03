from gensim.utils import simple_preprocess
from itertools import chain, islice
from datasets import load_dataset
import regex as re


class Corpora:
    def __init__(self, min_token_len, max_token_len, min_sentence_len):
        self.corpora = []
        self.min_token_len = min_token_len
        self.max_token_len = max_token_len
        self.min_sentence_len = min_sentence_len

    def add_corpus(self, name, source, fraction, is_large):
        self.corpora.append(Corpus(name, source, fraction, is_large,
                                   self.min_token_len, self.max_token_len, self.min_sentence_len))

    def __iter__(self):
        for sentence in chain.from_iterable(corpus.get_texts() for corpus in self.corpora):
            yield list(sentence['text'])


class Corpus:
    def __init__(self, name, source, fraction, is_large, min_token_len, max_token_len, min_sentence_len):
        self.name = name
        self.fraction = fraction
        self.is_large = is_large
        self.corpus = self.load_corpus(source, min_token_len, max_token_len, min_sentence_len)

    def load_corpus(self, source, min_token_len, max_token_len, min_sentence_len):
        corpus = []
        if self.is_large and source.name == 'huggingface':
            corpus = load_hg_dataset(self.name, min_token_len, max_token_len, min_sentence_len)
        else:
            self.load_local_corpus(source, corpus, min_token_len, max_token_len)
        return corpus

    def get_texts(self):
        if self.is_large and 0 < self.fraction < 1.0:
            return islice(self.corpus, int(self.corpus.info.splits['train'].num_examples * self.fraction))
        else:
            return self.corpus

    def load_local_corpus(self, source, corpus, min_token_len, max_token_len):
        if source.exists():
            files = [f for f in source.iterdir()]
            for file in files:
                if file.is_file():
                    with file.open('r') as f:
                        corpus.append({'text':
                                           preprocess_str({'text': f.read()}, min_token_len, max_token_len)['text']})
                elif file.is_dir():
                    self.load_local_corpus(file, corpus, min_token_len, max_token_len)


def load_hg_dataset(name, min_token_len, max_token_len, min_sentence_len):
    corpus = load_dataset('large_spanish_corpus', name=name, split='train', streaming=True)
    corpus = corpus.map(lambda row: preprocess_str(row, min_token_len, max_token_len))
    corpus = corpus.filter(lambda row: len(row['text']) > min_sentence_len)
    return corpus


def preprocess_str(string, min_token_len, max_token_len):
    string['text'] = re.sub(r'[^ \nA-Za-zÀ-ÖØ-öø-ÿ/]+', '', string['text'])
    string['text'] = simple_preprocess(string['text'], deacc=True, min_len=min_token_len, max_len=max_token_len)
    return string
