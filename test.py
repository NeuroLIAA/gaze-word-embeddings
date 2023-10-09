import pandas as pd
import numpy as np
from gensim.models import Word2Vec


def test(model_path, wa_file):
    model_file = model_path / f'{model_path.name}.model'
    if not model_file.exists():
        raise ValueError('The specified models does not exist')
    if not wa_file.exists():
        raise ValueError('The specified words association file does not exist')
    model = Word2Vec.load(str(model_file))
    words_associations = pd.read_csv(wa_file, index_col=0)
    words = words_associations.index
    frequency = answers_frequency(words_associations)
    freq_similarity_pairs = add_model_similarity(frequency, model)
    similarities_df = words_associations.apply(lambda answers: similarities(model, words, answers))
    save_path = model_path / 'test'
    save_path.mkdir(exist_ok=True)
    similarities_df.to_csv(save_path / f'{wa_file.stem}.csv')
    freq_similarity_pairs.to_csv(save_path / f'{wa_file.stem}_freq.csv')
    return similarities_df


def add_model_similarity(freq, model):
    words_pairs = []
    for cue in freq:
        cue_answers = freq[cue]
        for answer in cue_answers.keys():
            words_pairs.append((cue, answer, cue_answers[answer], word_similarity(model, cue, answer)))

    words_pairs = pd.DataFrame(words_pairs, columns=['cue', 'answer', 'freq', 'similarity'])
    return words_pairs


def similarities(model, words, answers):
    for i, answer in enumerate(answers):
        answers[i] = word_similarity(model, words[i], answer)
    return answers


def word_similarity(model, word, answer):
    if answer is None or answer not in model.wv or word not in model.wv:
        return np.nan
    return model.wv.similarity(word, answer)


def answers_frequency(words_associations):
    return {word: words_associations.loc[word].value_counts(normalize=True) for word in words_associations.index}
