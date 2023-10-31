from corpora import Corpora
from gensim.models import Word2Vec
from pathlib import Path


def save_model(model, fraction, save_path):
    model_name = f'{save_path.name}.model'
    if fraction < 1.0:
        save_path = save_path / f'{int(fraction * 100)}'
    save_path.mkdir(exist_ok=True, parents=True)
    model.save(str(save_path / model_name))


def train(corpora, source, fraction, repeats, min_token_len, max_token_len, min_sentence_len,
          vector_size, window_size, min_count, save_path):
    corpora = corpora.split('+')
    corpora = load_corpora(corpora, source, fraction, repeats, min_token_len, max_token_len, min_sentence_len)
    model = Word2Vec(sentences=corpora, vector_size=vector_size, window=window_size, min_count=min_count, workers=-1)
    save_model(model, fraction, save_path)
    return model


def load_corpora(corpora, source, fraction, repeats, min_token_len, max_token_len, min_sentence_len):
    training_corpora = Corpora(min_token_len, max_token_len, min_sentence_len)
    for corpus in corpora:
        is_large = 'texts' not in corpus
        source = Path(corpus) if not is_large else Path(source)
        training_corpora.add_corpus(corpus, source, fraction, repeats, is_large)
    return training_corpora
