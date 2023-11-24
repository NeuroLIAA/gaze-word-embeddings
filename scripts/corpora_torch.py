from gensim.utils import simple_preprocess
from itertools import chain, islice
from datasets import load_dataset
from pathlib import Path
from torch.utils.data import IterableDataset
import logging
import regex as re
import numpy as np
import torch

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


class Corpora(IterableDataset):
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, min_token_len, max_token_len, min_sentence_len, neg_samples=20, window_size=5, downsample=1e-3):
        super(Corpora).__init__()
        self.corpora = []
        self.min_token_len = min_token_len
        self.max_token_len = max_token_len
        self.min_sentence_len = min_sentence_len
        self.neg_samples = neg_samples
        self.window_size = window_size
        self.downsample = downsample

        self.neg_pos = 0
        self.negatives = []
        self.discards = []
        self.word2id = {}
        self.id2word = {}
        self.word_frequency = {}
        self.sentences_count = 0
        self.word_count = 0

    def add_corpus(self, name, source, fraction, repeats):
        if source == 'remote':
            repeats = 1
        for _ in range(repeats):
            self.corpora.append(Corpus(name, source, fraction,
                                       self.min_token_len, self.max_token_len, self.min_sentence_len))

    def get_size(self):
        for corpus in self.corpora:
            print(f'{corpus.name}: {corpus.size // (1024 * 1024)}MB, {corpus.num_sentences} sentences')

    def build_vocab(self, min_count):
        word_frequency = {}
        logging.info(f'Building vocabulary')
        last_logged = 0
        for sentence in chain.from_iterable(corpus.get_texts() for corpus in self.corpora):
            for word in sentence['text']:
                if len(word) > 0:
                    word_frequency[word] = word_frequency.get(word, 0) + 1
                    self.word_count += 1
            self.sentences_count += 1
            processed = int(self.sentences_count / self.__len__() * 100)
            if processed % 5 == 0 and processed != last_logged:
                logging.info(f'Processed {self.sentences_count}/{self.__len__()} ({processed}%) sentences')
                last_logged = processed

        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        self.init_table_negatives()
        self.init_table_discards()
        logging.info(f'Number of words: {len(self.word2id)}')

    def init_table_discards(self):
        f = np.array(list(self.word_frequency.values())) / self.word_count
        self.discards = np.sqrt(self.downsample / f) + (self.downsample / f)

    def init_table_negatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * Corpora.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def get_negatives(self, size):
        if self.neg_pos + size > len(self.negatives):
            np.random.shuffle(self.negatives)
            self.neg_pos = 0
        res = self.negatives[self.neg_pos:self.neg_pos + size]
        self.neg_pos += size
        return res

    def __len__(self):
        return sum(corpus.num_sentences for corpus in self.corpora)

    def __iter__(self):
        for sentence in chain.from_iterable(corpus.get_texts() for corpus in self.corpora):
            words = list(sentence['text'])
            if len(words) > 0:
                words_ids = [self.word2id[word] for word in words if word in self.word2id and
                             np.random.rand() < self.discards[self.word2id[word]]]
                reduced_window = np.random.randint(1, self.window_size + 1)
                yield [(u, v, self.get_negatives(self.neg_samples)) for i, u in enumerate(words_ids) for j, v in
                             enumerate(words_ids[max(i - reduced_window, 0):i + reduced_window]) if u != v]

    @staticmethod
    def collate(batches):
        all_u = np.array([u for batch in batches for u, _, _ in batch if len(batch) > 0])
        all_v = np.array([v for batch in batches for _, v, _ in batch if len(batch) > 0])
        all_neg_v = np.array([neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0])

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)


class Corpus:
    def __init__(self, name, source, fraction, min_token_len, max_token_len, min_sentence_len):
        self.name = name
        self.fraction = fraction
        self.is_remote = source == 'remote'
        self.size = 0
        self.num_sentences = 0
        self.corpus = self.load_corpus(min_token_len, max_token_len, min_sentence_len)

    def load_corpus(self, min_token_len, max_token_len, min_sentence_len):
        corpus = []
        if self.is_remote:
            corpus = self.load_hg_dataset(self.name, min_token_len, max_token_len, min_sentence_len)
        else:
            self.load_local_corpus(Path(self.name), corpus, min_token_len, max_token_len)
        return corpus

    def get_texts(self):
        if self.is_remote and 0 < self.fraction < 1.0:
            return islice(self.corpus, self.num_sentences)
        else:
            return self.corpus

    def load_local_corpus(self, data_path, corpus, min_token_len, max_token_len):
        if data_path.exists():
            files = [f for f in data_path.iterdir()]
            for file in files:
                if file.is_file():
                    with file.open('r') as f:
                        sentences = [{'text': preprocess_str({'text': sentence}, min_token_len, max_token_len)['text']}
                                     for sentence in f.read().split('.')]
                        self.size += sum(len(sentence['text']) for sentence in sentences)
                        self.num_sentences += len(sentences)
                        corpus.extend(sentences)
                elif file.is_dir():
                    self.load_local_corpus(file, corpus, min_token_len, max_token_len)

    def load_hg_dataset(self, name, min_token_len, max_token_len, min_sentence_len):
        corpus = load_dataset('large_spanish_corpus', name=name, split='train', streaming=True)
        corpus = corpus.map(lambda row: preprocess_str(row, min_token_len, max_token_len))
        corpus = corpus.filter(lambda row: len(row['text']) > min_sentence_len)
        self.size = int(corpus.info.splits['train'].num_bytes * self.fraction)
        self.num_sentences = int(corpus.info.splits['train'].num_examples * self.fraction)
        return corpus


def preprocess_str(string, min_token_len, max_token_len):
    deaccent_map = str.maketrans(DEACCENT_MAP)
    string['text'] = re.sub(r'[^ \nA-Za-zÀ-ÖØ-öø-ÿ/]+', '', string['text'])
    string['text'] = string['text'].translate(deaccent_map)
    string['text'] = simple_preprocess(string['text'], min_len=min_token_len, max_len=max_token_len)
    return string
