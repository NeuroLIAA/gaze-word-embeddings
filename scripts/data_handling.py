from collections import Counter, OrderedDict
from torch.utils.data import DataLoader
from scripts.utils import get_words_in_corpus
from scripts.vocabulary import Vocabulary
from functools import partial
from itertools import islice
from tqdm import tqdm
import numpy as np
import torch


def build_downsample_distribution(word_freq, total_words, downsample_factor, base_vocab_tokensids):
    frequency = np.array(list(word_freq.values())) / total_words
    frequency = np.sqrt(downsample_factor / frequency) + (downsample_factor / frequency)
    # Insert <unk> index and make it so that it is always discarded
    frequency = np.insert(frequency, 0, 0.0)
    if len(base_vocab_tokensids) > 0:
        for base_vocab_token in base_vocab_tokensids:
            frequency[base_vocab_token] = 0.0
    return frequency


def build_vocab(corpora, min_count, max_vocab_size=None, words_in_stimuli=None):
    word_freq = Counter()
    print('Building vocabulary')
    for tokens in tqdm(corpora):
        word_freq.update(tokens['text'])
    total_words = sum(word_freq.values())
    if max_vocab_size is not None:
        word_freq = word_freq.most_common(max_vocab_size)
    else:
        word_freq = word_freq.items()
    
    word_freq = OrderedDict(sorted(word_freq, key=lambda x: x[1], reverse=True))
    vocabulary = Vocabulary(word_freq, min_freq=min_count)
    # Keep only words with frequency >= min_count
    word_freq = OrderedDict(islice(word_freq.items(), len(vocabulary)))
    base_vocab_tokens = add_base_vocab(words_in_stimuli, vocabulary, word_freq)
    return vocabulary, word_freq, total_words, base_vocab_tokens


def add_base_vocab(words_in_stimuli, vocabulary, word_freq, eps=1e-8):
    base_vocab_tokens = []
    if words_in_stimuli is not None:
        for word in words_in_stimuli:
            if word not in vocabulary:
                vocabulary.add_word(word)
                base_vocab_tokens.append(word)
                word_freq[word] = eps
    return base_vocab_tokens


def get_vocab(corpora, min_count, words_in_stimuli, is_baseline, vocab_savepath, max_vocab_size=None):
    if vocab_savepath.exists():
        print('Loading vocabulary from checkpoint')
        vocabulary, word_freq, total_words, base_vocab_tokens = torch.load(vocab_savepath, weights_only=False).values()
        if not is_baseline:
            _, word_freq_ft, total_words, base_vocab_tokens = build_vocab(corpora, min_count, max_vocab_size=max_vocab_size,
                                                                             words_in_stimuli=words_in_stimuli)
            word_freq.update(word_freq_ft)
    else:
        vocabulary, word_freq, total_words, base_vocab_tokens = build_vocab(corpora, min_count, max_vocab_size=max_vocab_size,
                                                                               words_in_stimuli=words_in_stimuli)
        if is_baseline:
            torch.save({'vocabulary': vocabulary, 'word_freq': word_freq, 'total_words': total_words,
                        'base_vocab_tokens': base_vocab_tokens}, vocab_savepath)
    return vocabulary, word_freq, total_words, base_vocab_tokens


def get_dataloader_and_vocab(corpora, min_count, n_negatives, downsample_factor, window_size, batch_size, train_fix,
                             stimuli_path, pretrained_path, save_path):
    words_in_stimuli = get_words_in_corpus(stimuli_path)
    vocab_savepath = save_path / 'vocab.pt' if not pretrained_path else pretrained_path / 'vocab.pt'
    vocabulary, word_freq, total_words, base_vocab_tokens = get_vocab(corpora, min_count, words_in_stimuli,
                                                                         pretrained_path is None, vocab_savepath)
    negative_samples_set = Samples(word_freq)
    downsample_table = build_downsample_distribution(word_freq, total_words, downsample_factor,
                                                     vocabulary(base_vocab_tokens))
    dataloader = DataLoader(
        corpora,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=partial(collate_fn,
                           words_mapping=lambda words: vocabulary(words),
                           window_size=window_size,
                           negative_samples=negative_samples_set,
                           downsample_table=downsample_table,
                           n_negatives=n_negatives,
                           predict_fix=train_fix),
        num_workers=8
    )
    return dataloader, vocabulary


def collate_fn(batch, words_mapping, window_size, negative_samples, downsample_table, n_negatives, predict_fix):
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
            word_fix = words_fix.pop(input_word_idx)
            batch_input.extend([word_id] * len(context_words))
            batch_output.extend(context_words)
            batch_negatives.extend([negative_samples.sample(n_negatives) for _ in range(len(context_words))])
            batch_fixations.extend(words_fix if predict_fix == 'output' else [word_fix] * len(context_words))

    batch_input = np.array(batch_input)
    batch_output = np.array(batch_output)
    batch_negatives = np.array(batch_negatives)
    batch_fixations = np.array(batch_fixations)
    return (torch.LongTensor(batch_input), torch.LongTensor(batch_output), torch.LongTensor(batch_negatives),
            torch.FloatTensor(batch_fixations))


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
