from functools import partial
import torch
from gensim.utils import simple_preprocess
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from itertools import islice
import numpy as np
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


def build_downsample_distribution(word_freq, total_words, downsample_factor):
    frequency = np.array(list(word_freq.values())) / total_words
    frequency = np.sqrt(downsample_factor / frequency) + (downsample_factor / frequency)
    # Insert <unk> index and make it so that it is always discarded
    frequency = np.insert(frequency, 0, 0.0)
    return frequency


def build_vocab(corpora, min_count):
    word_freq = Counter()
    for tokens in corpora:
        word_freq.update(tokens['text'])
    total_words = sum(word_freq.values())
    word_freq = OrderedDict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))
    vocabulary = vocab(word_freq, min_freq=min_count)
    # Keep only words with frequency >= min_count
    word_freq = OrderedDict(islice(word_freq.items(), len(vocabulary)))
    vocabulary.insert_token('<unk>', 0)
    vocabulary.set_default_index(vocabulary['<unk>'])

    return vocabulary, word_freq, total_words


def get_dataloader_and_vocab(corpora, min_count, n_negatives, downsample_factor, window_size, batch_size):
    vocabulary, word_freq, total_words = build_vocab(corpora, min_count)
    negative_samples_set = Samples(word_freq)
    downsample_table = build_downsample_distribution(word_freq, total_words, downsample_factor)

    dataloader = DataLoader(
        corpora,
        batch_size=batch_size,
        collate_fn=partial(collate_fn,
                           words_mapping=lambda words: vocabulary(words),
                           window_size=window_size,
                           negative_samples=negative_samples_set,
                           downsample_table=downsample_table,
                           n_negatives=n_negatives),
    )

    return dataloader, vocabulary


def collate_fn(batch, words_mapping, window_size, negative_samples, downsample_table, n_negatives):
    rnd_generator = np.random.default_rng()
    batch_input, batch_output, batch_negatives, batch_fixations = [], [], [], []
    for sentence in batch:
        words_ids, words_fix = words_mapping(sentence['text']), sentence['fix_dur']
        words, fixs = [], []
        for i, word_id in enumerate(words_ids):
            if rnd_generator.random() < downsample_table[word_id]:
                words.append(word_id)
                fixs.append(words_fix[i])
        reduced_window = rnd_generator.integers(1, window_size + 1)
        for idx, word_id in enumerate(words):
            context_words = words[max(idx - reduced_window, 0): idx + reduced_window]
            words_fix = fixs[max(idx - reduced_window, 0): idx + reduced_window]
            input_word_idx = idx if idx < reduced_window else reduced_window
            context_words.pop(input_word_idx)
            words_fix.pop(input_word_idx)
            batch_input.extend([word_id] * len(context_words))
            batch_output.extend(context_words)
            batch_negatives.extend([negative_samples.sample(n_negatives) for _ in range(len(context_words))])
            batch_fixations.extend(words_fix)

    batch_input = np.array(batch_input)
    batch_output = np.array(batch_output)
    batch_negatives = np.array(batch_negatives)
    batch_fixations = np.array(batch_fixations)

    return (torch.LongTensor(batch_input), torch.LongTensor(batch_output), torch.LongTensor(batch_negatives),
            torch.LongTensor(batch_fixations))


class Samples:
    def __init__(self, word_freq, size=1e8):
        self.current_pos = 0
        self.rng = np.random.default_rng()
        self.samples = self.build_samples(word_freq, size)

    def build_samples(self, word_freq, size):
        # This is highly dependent on word_freq having the same order as vocabulary
        sqrt_freq = np.array(list(word_freq.values())) ** 0.5
        ratio = sqrt_freq / sum(sqrt_freq)
        count = np.round(ratio * size)
        samples = []
        for wid, c in enumerate(count, 1):
            samples += [wid] * int(c)
        samples = np.array(samples)
        self.rng.shuffle(samples)
        return samples

    def sample(self, num_samples):
        if self.current_pos + num_samples > len(self.samples):
            self.rng.shuffle(self.samples)
            self.current_pos = 0
        samples = self.samples[self.current_pos:self.current_pos + num_samples]
        self.current_pos += num_samples
        return samples


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
        return self.corpora[idx]

    def __iter__(self):
        for sentence in self.corpora:
            yield sentence


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
            data = load_dataset('large_spanish_corpus', name=self.name, split=f'train[:{fraction}%]', num_proc=12)
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
    string['text'] = [token for token in string['text'] if re.match(r'^[A-Za-zñ]+$', token)]
    if 'fix_dur' not in string:
        string['fix_dur'] = [0] * len(string['text'])
    return string
