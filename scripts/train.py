from scripts.corpora import Corpora
from gensim.models import Word2Vec


def train(corpora_labels, data_sources, fraction, repeats, skip_gram, negative_samples, epochs,
          min_token_len, max_token_len, min_sentence_len, vector_size, window_size, min_count, save_path):
    print(f'Beginning training with corpora {corpora_labels} ({int(fraction * 100)}% of baseline corpus)')
    corpora = load_corpora(corpora_labels, data_sources, fraction, repeats,
                           min_token_len, max_token_len, min_sentence_len)
    model = Word2Vec(sentences=corpora, sg=skip_gram, vector_size=vector_size, window=window_size, min_count=min_count,
                     negative=negative_samples, epochs=epochs, workers=-1)
    model_name, save_path = get_path(save_path, corpora_labels, data_sources)
    save_path.mkdir(exist_ok=True, parents=True)
    model.save(str(save_path / f'{model_name}.model'))
    print(f'Training completed. Model saved at {save_path}')
    return model


def load_corpora(corpora_labels, data_sources, fraction, repeats, min_token_len, max_token_len, min_sentence_len):
    training_corpora = Corpora(min_token_len, max_token_len, min_sentence_len)
    for corpus, source in zip(corpora_labels, data_sources):
        training_corpora.add_corpus(corpus, source, fraction, repeats)
    return training_corpora


def get_path(save_path, corpora_labels, data_sources):
    model_name = corpora_labels[-1] if 'local' in data_sources else 'baseline'
    save_path = save_path / model_name
    return model_name, save_path
