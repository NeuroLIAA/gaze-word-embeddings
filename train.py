from corpora import Corpora
from gensim.models import Word2Vec, callbacks
from pathlib import Path


class TrainLogger(callbacks.CallbackAny2Vec):
    def __init__(self, model_name, fraction=1.0):
        self.model_name = model_name
        self.fraction = fraction
        self.epoch = 0

    def on_train_begin(self, model):
        print(f'Beginning training of model {self.model_name} with {int(self.fraction * 100)}% of baseline corpus')

    def on_train_end(self, model):
        print(f'Finished training model {self.model_name} with {int(self.fraction * 100)}% of baseline corpus')


def train(corpora, source, fraction, repeats, min_token_len, max_token_len, min_sentence_len,
          vector_size, window_size, min_count, model_name, save_path):
    corpora = corpora.split('+')
    corpora = load_corpora(corpora, source, fraction, repeats, min_token_len, max_token_len, min_sentence_len)
    train_logger = TrainLogger(model_name, fraction)
    model = Word2Vec(sentences=corpora, vector_size=vector_size, window=window_size, min_count=min_count,
                     callbacks=[train_logger], workers=-1)
    save_path.mkdir(exist_ok=True, parents=True)
    model.save(str(save_path / f'{model_name}.model'))
    return model


def load_corpora(corpora, source, fraction, repeats, min_token_len, max_token_len, min_sentence_len):
    training_corpora = Corpora(min_token_len, max_token_len, min_sentence_len)
    for corpus in corpora:
        is_large = 'texts' not in corpus
        source = Path(corpus) if not is_large else Path(source)
        training_corpora.add_corpus(corpus, source, fraction, repeats, is_large)
    return training_corpora
