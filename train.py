from corpora import Corpora
from gensim.models import Word2Vec
from pathlib import Path


def train(corpora_labels, data_sources, fraction, repeats, min_token_len, max_token_len, min_sentence_len,
          vector_size, window_size, min_count, model_name, save_path):
    print(f'Beginning training with corpora {corpora_labels} ({int(fraction * 100)}% of baseline corpus)')
    corpora = load_corpora(corpora_labels, data_sources, fraction, repeats,
                           min_token_len, max_token_len, min_sentence_len)
    model = Word2Vec(sentences=corpora, vector_size=vector_size, window=window_size, min_count=min_count, workers=-1)
    save_path = get_path(save_path, corpora_labels)
    save_path.mkdir(exist_ok=True, parents=True)
    model.save(str(save_path / f'{model_name}.model'))
    print(f'Training completed. Model saved at {save_path}')
    return model


def load_corpora(corpora_labels, data_sources, fraction, repeats, min_token_len, max_token_len, min_sentence_len):
    training_corpora = Corpora(min_token_len, max_token_len, min_sentence_len)
    for corpus, source in zip(corpora_labels, data_sources):
        training_corpora.add_corpus(corpus, source, fraction, repeats)
    return training_corpora


def get_path(save_path, corpora_labels):
    if len(corpora_labels) > 1:
        save_path = save_path / corpora_labels[-1]
    else:
        save_path = save_path / 'baseline'
    return save_path
