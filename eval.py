

def answers_frequency(words_associations):
    return {word: words_associations.loc[word].value_counts(normalize=True) for word in words_associations.index}
