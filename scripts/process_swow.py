import pandas as pd
import numpy as np
from pathlib import Path
from scripts.utils import get_words_in_corpus
from sklearn.model_selection import train_test_split

NON_LATIN_REGEX = '[^ \nA-Za-zá-úñ]+'
INVALID_INBETWEEN_REGEX = '[A-Za-zá-úñ]+.*?[^\sA-Za-zá-úñ][A-Za-zá-úñ]+'


def load_swow(wa_file, wf_file, stimuli_path, data_set, min_n, seed):
    wa_set_file = Path(f'{wa_file[:-4]}_{data_set}.csv')
    if not wa_set_file.exists():
        process_swow(wa_file, wf_file, stimuli_path, min_n, seed)
    return pd.read_csv(wa_set_file)


def process_swow(swow_file, words_freq, stimuli_path, min_n, seed):
    words_in_stimuli = get_words_in_corpus(stimuli_path)
    swow = pd.read_csv(swow_file, sep='\t')
    swow.drop(columns=['N'], inplace=True)
    swow.rename(columns={'response': 'answer', 'R1': 'n', 'R1.Strength': 'freq'}, inplace=True)
    swow.drop_duplicates(subset=['cue', 'answer'], inplace=True)

    for keyword in ['cue', 'answer']:
        swow = remove_composed_words(swow, keyword)
        swow = remove_regex(swow, keyword, NON_LATIN_REGEX)
        swow = remove_regex(swow, keyword, INVALID_INBETWEEN_REGEX)
        swow = remove_short_and_long_words(swow, keyword, 3, 20)
        swow[keyword] = swow[keyword].str.lower()
        if keyword == 'cue':
            swow = remove_words_not_in_espal(swow, keyword, words_freq)

    swow = compute_mean_n(swow)
    swow = swow[swow['n'] >= min_n]
    swow_val, swow_test = val_test_split(swow, words_in_stimuli, test_size=0.5, random_state=seed)
    swow_val.to_csv(f'{swow_file[:-4]}_val.csv', index=False)
    swow_test.to_csv(f'{swow_file[:-4]}_test.csv', index=False)


def val_test_split(swow, words_in_stimuli, test_size=0.5, random_state=42):
    swow_in_stimuli = swow[(swow['cue'].isin(words_in_stimuli)) & (swow['answer'].isin(words_in_stimuli))]
    swow_off_stimuli = swow[(~swow['cue'].isin(words_in_stimuli)) & (~swow['answer'].isin(words_in_stimuli))]
    swow_in_val, swow_in_test = train_test_split(swow_in_stimuli, test_size=test_size, random_state=random_state)
    swow_off_val, swow_off_test = train_test_split(swow_off_stimuli, test_size=test_size, random_state=random_state)
    swow_val = pd.concat([swow_in_val, swow_off_val])
    swow_test = pd.concat([swow_in_test, swow_off_test])
    swow_val = swow_val.sample(frac=1, random_state=42).reset_index(drop=True)
    swow_test = swow_test.sample(frac=1, random_state=42).reset_index(drop=True)
    return swow_val, swow_test


def compute_mean_n(swow):
    original_cues = swow['cue'].unique().copy()
    swow_renamed = swow.rename(columns={'answer': 'cue', 'cue': 'answer'})
    swow_merged = pd.merge(swow, swow_renamed, on='cue').drop(columns=['freq_x', 'freq_y'])
    same_word_pairs = swow_merged[swow_merged['answer_x'] == swow_merged['answer_y']].copy()
    # Average the number of responses for the same word pairs
    same_word_pairs['n'] = (same_word_pairs['n_x'] + same_word_pairs['n_y']) // 2
    same_word_pairs.drop(columns=['n_x', 'n_y', 'answer_y'], inplace=True)
    same_word_pairs.rename(columns={'answer_x': 'answer'}, inplace=True)
    # Compute frequency of answer with new n
    swow_merged = pd.merge(swow, same_word_pairs, on=['cue', 'answer'], how='left')
    swow_merged['n'] = swow_merged['n_y'].fillna(swow_merged['n_x'])
    swow_merged.drop(columns=['n_x', 'n_y', 'freq'], inplace=True)
    swow_merged['n'] = swow_merged['n'].astype(int)
    swow_merged['freq'] = swow_merged.groupby('cue')['n'].transform(lambda x: x / x.sum())
    # Get rid of repeated word pairs and keep the original cues
    swow_merged[['cue', 'answer']] = pd.DataFrame(np.sort(swow_merged[['cue', 'answer']], axis=1),
                                                  index=swow_merged.index)
    swow_merged.drop_duplicates(['cue', 'answer'], inplace=True)
    return swow_merged[swow_merged['cue'].isin(original_cues)]


def remove_composed_words(df, column):
    return df[~df[column].str.contains(' ')]


def remove_regex(df, column, regex):
    return df[~df[column].str.contains(regex)]


def remove_short_and_long_words(df, column, min_len, max_len):
    return df[(df[column].str.len() >= min_len) & (df[column].str.len() <= max_len)]


def remove_words_not_in_espal(df, column, words_freq):
    return df[df[column].isin(words_freq['word'])]