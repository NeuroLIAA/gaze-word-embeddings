import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scripts.utils import apply_threshold, filter_low_frequency_answers
from test import report_similarity, compare_to_baseline


def plot_distance_to_gt(model_basename, distances_to_embeddings, sim_threshold, gt_threshold,
                        save_path, error_bars=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    title = f'Distance to ground truth embeddings ({model_basename}) (lower is better)'
    diff_df, se_df = pd.DataFrame(), pd.DataFrame()
    for model_name in distances_to_embeddings:
        model_distances = distances_to_embeddings[model_name]
        model_distances[['sim']] = apply_threshold(model_distances[['sim']], sim_threshold)
        model_distances[['sim_gt']] = apply_threshold(model_distances[['sim_gt']], gt_threshold)
        model_distances['diff'] = abs(model_distances['sim'] - model_distances['sim_gt'])
        mean_diff = model_distances.groupby('in_stimuli')['diff'].mean()
        std_diff = model_distances.groupby('in_stimuli')['diff'].std()
        se_diff = std_diff / np.sqrt(model_distances.shape[0])
        diff_df = pd.concat([diff_df, mean_diff.to_frame(model_name)], axis=1)
        se_df = pd.concat([se_df, se_diff.to_frame(model_name)], axis=1)
    if error_bars:
        diff_df.plot.bar(xlabel='Words present in stimuli', yerr=se_df, capsize=4, ax=ax)
    else:
        diff_df.plot.bar(xlabel='Words present in stimuli', ax=ax)
    ax.set_title(title)
    ax.legend()
    ax.set_ylabel('Similarity difference with SWOW-RP embeddings')
    plt.savefig(save_path / f'{title}.png')
    plt.show()


def plot_freq_to_sim(basename, similarities_to_answers, save_path, min_appearances):
    fig, ax = plt.subplots(figsize=(15, 6))
    title = f'Human frequency to model similarity ({basename})'
    for model_name in similarities_to_answers:
        model_sim_to_answers = similarities_to_answers[model_name]
        wa_freq_sim_to_plot = filter_low_frequency_answers(model_sim_to_answers, min_appearances)
        ax.scatter(wa_freq_sim_to_plot['similarity'], wa_freq_sim_to_plot['freq'], label=model_name)
    ax.set_xlabel('Model similarity')
    ax.set_ylabel('Human frequency of answer')
    ax.set_title(title)
    ax.legend()
    plt.savefig(save_path / 'freq_to_sim.png')
    plt.show()


def plot_similarity(model_basename, similarities_to_subjs, sim_threshold, save_path, sort_by='texts', error_bars=True):
    if 'baseline' not in similarities_to_subjs:
        print('No baseline model found. Skipping similarity plots')
        return
    for axis, comparable in zip([0, 1], ['subjects', 'cues']):
        fig, ax = plt.subplots(figsize=(25, 15))
        title = f'Avg. similarity to {comparable} answers (baseline: {model_basename}) (higher is better)'
        print(f'\n------{title}------')
        mean_similarities, se_similarities = pd.DataFrame(), pd.DataFrame()
        for model_name in similarities_to_subjs:
            model_sim_to_subjs = apply_threshold(similarities_to_subjs[model_name], sim_threshold)
            mean_subj_sim, se_subj_sim = report_similarity(model_name, model_sim_to_subjs, axis)
            mean_similarities = pd.concat([mean_similarities, mean_subj_sim], axis=1)
            se_similarities = pd.concat([se_similarities, se_subj_sim], axis=1)
        mean_similarities, se_similarities = compare_to_baseline(mean_similarities, se_similarities)
        mean_similarities = mean_similarities.sort_values(by=sort_by, ascending=False)
        se_similarities = se_similarities.reindex(mean_similarities.index)
        if error_bars:
            mean_similarities.plot.bar(yerr=se_similarities, capsize=4, ax=ax)
        else:
            mean_similarities.plot.bar(ax=ax)
        ax.set_title(title, fontsize=15)
        ax.legend(fontsize=15)
        ax.set_ylabel('Similarity diff. to baseline', fontsize=15)
        plt.savefig(save_path / f'{title}.png')
        plt.show()
