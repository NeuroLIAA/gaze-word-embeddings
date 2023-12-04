from gensim.utils import simple_preprocess
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
import regex as re

DEACCENT_MAP = {'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
                'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u'}


class Corpora(Dataset):

    def __init__(self, min_token_len, max_token_len, min_sentence_len):
        super(Corpora).__init__()
        self.corpora = None
        self.size = {}
        self.num_sentences = {}
        self.min_token_len = min_token_len
        self.max_token_len = max_token_len
        self.min_sentence_len = min_sentence_len

    def add_corpus(self, name, source, fraction, repeats):
        if source == 'remote':
            repeats = 1
        for _ in range(repeats):
            corpus = Corpus(name, source, fraction,
                            self.min_token_len, self.max_token_len, self.min_sentence_len)
            self.corpora = corpus.data if self.corpora is None else concatenate_datasets([self.corpora, corpus.data])
            self.size[name] = self.size.get(name, 0) + corpus.size
            self.num_sentences[name] = self.num_sentences.get(name, 0) + corpus.num_sentences

    def print_size(self):
        total_size, total_sentences = 0, 0
        for corpus in self.size:
            print(f'{corpus}: {self.size[corpus] // (1024 * 1024)}MB, {self.num_sentences[corpus]} sentences')
            total_size += self.size[corpus]
            total_sentences += self.num_sentences[corpus]
        print(f'Total: {total_size // (1024 * 1024)}MB, {total_sentences} sentences')

    def __len__(self):
        return self.corpora.num_rows

    def __getitem__(self, idx):
        return self.corpora[idx]['text']

    def __iter__(self):
        for sentence in self.corpora:
            yield list(sentence['text'])


class Corpus:
    def __init__(self, name, source, fraction, min_token_len, max_token_len, min_sentence_len):
        self.name = name
        self.is_remote = source == 'remote'
        self.size = 0
        self.num_sentences = 0
        self.data = self.load_corpus(min_token_len, max_token_len, min_sentence_len, fraction)

    def load_corpus(self, min_token_len, max_token_len, min_sentence_len, fraction):
        if self.is_remote:
            fraction = int(fraction * 100)
            data = load_dataset('large_spanish_corpus', name=self.name, split=f'train[:{fraction}%]')
        else:
            data = load_dataset(self.name)['train']
        self.size = data.info.size_in_bytes
        data = data.map(lambda row: preprocess_str(row, min_token_len, max_token_len), num_proc=12)
        data = data.filter(lambda row: min_sentence_len < len(row['text']), num_proc=12)
        self.num_sentences = data.num_rows
        return data


def preprocess_str(string, min_token_len, max_token_len):
    deaccent_map = str.maketrans(DEACCENT_MAP)
    string['text'] = string['text'].translate(deaccent_map)
    string['text'] = simple_preprocess(string['text'], min_len=min_token_len, max_len=max_token_len)
    string['text'] = [token for token in string['text'] if re.match(r'^[A-Za-z]+$', token)]
    return string
