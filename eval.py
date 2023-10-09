import pandas as pd
import numpy as np


def answers_frequency(words_associations):
    return {word: words_associations.loc[word].value_counts(normalize=True) for word in words_associations.index}


def add_model_similarity(freq, model):
    words_pairs = []
    for cue in freq:
        cue_answers = freq['cue']
        for answer in cue_answers:
            words_pairs.append((cue, answer, cue_answers[answer], word_similarity(model, cue, answer)))

    words_pairs = pd.DataFrame(words_pairs, columns=['cue', 'answer', 'freq', 'similarity'])
    return words_pairs


def word_similarity(model, word, answer):
    if answer is None or answer not in model.wv or word not in model.wv:
        return np.nan
    return model.wv.similarity(word, answer)
