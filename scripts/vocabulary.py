class Vocabulary:
    def __init__(self, word_freq, min_freq=1):
        self.word2idx = {}
        self.idx2word = []
        self.word_freq = word_freq
        self.min_freq = min_freq
        self.build_vocab()

    def build_vocab(self):
        self.add_word('<unk>')
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq:
                self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

    def __len__(self):
        return len(self.idx2word)

    def __call__(self, words):
        if isinstance(words, list):
            return [self.word2idx.get(word, self.word2idx['<unk>']) for word in words]
        return self.word2idx.get(words, self.word2idx['<unk>'])

    def __contains__(self, word):
        return word in self.word2idx

    def get_itos(self):
        return self.idx2word
    
    def get_stoi(self):
        return self.word2idx

    def get_word(self, idx):
        return self.idx2word[idx]
