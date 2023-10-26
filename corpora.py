from gensim.utils import simple_preprocess
from itertools import chain, islice
from datasets import load_dataset
import regex as re

DEACCENT_MAP = {'À': 'A', 'Á': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'A',
                'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a',
                'È': 'E', 'É': 'E', 'Ê': 'E', 'Ë': 'E',
                'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
                'Í': 'I', 'Ì': 'I', 'Î': 'I', 'Ï': 'I',
                'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
                'Ò': 'O', 'Ó': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'O',
                'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o',
                'Ù': 'U', 'Ú': 'U', 'Û': 'U', 'Ü': 'U',
                'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
                'Ç': 'C', 'ç': 'c'}


class Corpora:
    def __init__(self, min_token_len, max_token_len, min_sentence_len):
        self.corpora = []
        self.min_token_len = min_token_len
        self.max_token_len = max_token_len
        self.min_sentence_len = min_sentence_len

    def add_corpus(self, name, source, fraction, repeats, is_large):
        if is_large or repeats == 1:
            self.corpora.append(Corpus(name, source, fraction, is_large,
                                       self.min_token_len, self.max_token_len, self.min_sentence_len))
        else:
            for _ in range(repeats):
                self.corpora.append(Corpus(name, source, fraction, is_large,
                                           self.min_token_len, self.max_token_len, self.min_sentence_len))

    def get_size(self):
        return {corpus.name: corpus.size for corpus in self.corpora}

    def __iter__(self):
        for sentence in chain.from_iterable(corpus.get_texts() for corpus in self.corpora):
            yield list(sentence['text'])


class Corpus:
    def __init__(self, name, source, fraction, is_large, min_token_len, max_token_len, min_sentence_len):
        self.name = name
        self.fraction = fraction
        self.is_large = is_large
        self.size = 0
        self.corpus = self.load_corpus(source, min_token_len, max_token_len, min_sentence_len)

    def load_corpus(self, source, min_token_len, max_token_len, min_sentence_len):
        corpus = []
        if self.is_large and source.name == 'huggingface':
            corpus = self.load_hg_dataset(self.name, min_token_len, max_token_len, min_sentence_len)
        else:
            self.load_local_corpus(source, corpus, min_token_len, max_token_len)
        return corpus

    def get_texts(self):
        if self.is_large and 0 < self.fraction < 1.0:
            return islice(self.corpus, self.size)
        else:
            return self.corpus

    def load_local_corpus(self, source, corpus, min_token_len, max_token_len):
        if source.exists():
            files = [f for f in source.iterdir()]
            for file in files:
                if file.is_file():
                    with file.open('r') as f:
                        sentences = [{'text': preprocess_str({'text': sentence}, min_token_len, max_token_len)['text']}
                                     for sentence in f.read().split('.')]
                        self.size += len(sentences)
                        corpus.extend(sentences)
                elif file.is_dir():
                    self.load_local_corpus(file, corpus, min_token_len, max_token_len)

    def load_hg_dataset(self, name, min_token_len, max_token_len, min_sentence_len):
        corpus = load_dataset('large_spanish_corpus', name=name, split='train', streaming=True)
        corpus = corpus.map(lambda row: preprocess_str(row, min_token_len, max_token_len))
        corpus = corpus.filter(lambda row: len(row['text']) > min_sentence_len)
        self.size = int(corpus.info.splits['train'].num_examples * self.fraction)
        return corpus


def preprocess_str(string, min_token_len, max_token_len):
    deaccent_map = str.maketrans(DEACCENT_MAP)
    string['text'] = re.sub(r'[^ \nA-Za-zÀ-ÖØ-öø-ÿ/]+', '', string['text'])
    string['text'] = string['text'].translate(deaccent_map)
    string['text'] = simple_preprocess(string['text'], min_len=min_token_len, max_len=max_token_len)
    return string
