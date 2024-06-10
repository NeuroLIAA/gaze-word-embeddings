import numpy as np
from matplotlib import pyplot as plt, colormaps
from scripts.utils import apply_threshold
from numpy import linspace


def plot_correlations(models_results, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for i, stimuli in enumerate(['in_stimuli', 'off_stimuli']):
        title = f'{"in" if stimuli == "in_stimuli" else "off"} stimuli words'
        data = [models_results[stimuli][model_name] for model_name in models_results[stimuli]]
        ax[i].boxplot(data, labels=models_results[stimuli].keys())
        ax[i].set_title(title)
        ax[i].set_xlabel('Model')
        ax[i].set_ylabel('Spearman correlation coefficient')
        plt.setp(ax[i].xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(save_path / 'correlations.png')
    plt.show()


def plot_distance_to_gt_across_thresholds(distances_to_embeddings, models_thresholds, save_path, error_bars=True):
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    percentiles = np.arange(0, 100, 5)
    for i, in_stimuli in enumerate([True, False]):
        title = (f'Distance to ground truth embeddings (lower is better) for words {"in" if in_stimuli else "not in"}'
                 f' stimuli')
        diff_dict = {model_name: [] for model_name in distances_to_embeddings}
        se_dict = {model_name: [] for model_name in distances_to_embeddings}
        gt_thresholds = models_thresholds['SWOW-RP']['in' if in_stimuli else 'off']
        for model_name in distances_to_embeddings:
            sim_thresholds = models_thresholds[model_name]['in' if in_stimuli else 'off']
            for sim_threshold, gt_threshold in zip(sim_thresholds, gt_thresholds):
                model_similarities = distances_to_embeddings[model_name].copy()
                model_similarities = model_similarities[model_similarities['in_stimuli'] == in_stimuli]
                model_similarities[['sim']] = apply_threshold(model_similarities[['sim']], sim_threshold)
                model_similarities[['sim_gt']] = apply_threshold(model_similarities[['sim_gt']], gt_threshold)
                model_similarities['diff'] = abs(model_similarities['sim'] - model_similarities['sim_gt'])
                mean_diff = model_similarities['diff'].mean()
                std_diff = model_similarities['diff'].std()
                se_diff = std_diff / np.sqrt(model_similarities.shape[0])
                diff_dict[model_name].append(mean_diff)
                se_dict[model_name].append(se_diff)
        for model_name in diff_dict:
            if error_bars:
                ax[i].errorbar(percentiles, diff_dict[model_name], yerr=se_dict[model_name], label=model_name)
            else:
                ax[i].plot(percentiles, diff_dict[model_name], label=model_name)
        ax[i].set_title(title)
        ax[i].set_xlabel('Threshold at percentile')
        ax[i].set_ylabel('Similarity difference with SWOW-RP embeddings')
        ax[i].legend()
    plt.savefig(save_path / f'distance_to_gt_across_thresholds.png')
    plt.show()


def plot_loss(loss_sg, loss_fix, model_name, save_path):
    plt.plot(loss_sg, label='W2V', alpha=0.7)
    plt.plot(loss_fix, label='Fix duration', alpha=0.7)
    plt.legend()
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} loss')
    plt.savefig(save_path / 'loss.png')


def plot_ppl(ppl, model_name, save_path):
    keys = list(ppl.keys())
    values = list(ppl.values())

    plt.figure(figsize=(10, 5))
    plt.plot(keys, values, marker='o')
    plt.title(model_name)
    plt.xlabel('Batch')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.savefig(save_path / 'ppl.png')


def similarity_distributions(models_similarities, save_path):
    num_models = len(models_similarities)
    colors = colormaps['Accent'](linspace(0, 1, num_models))
    fig_hist, ax_hist = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    hist_bins = np.arange(-0.4, 0.8, 0.01)
    models_thresholds = {model_name: {} for model_name in models_similarities}
    models_thresholds['SWOW-RP'] = {}
    for i, model_name in enumerate(models_similarities):
        for j, in_stimuli in enumerate([True, False]):
            model_similarities = models_similarities[model_name].copy()
            model_similarities = model_similarities[model_similarities['in_stimuli'] == in_stimuli]
            models_thresholds[model_name]['in' if in_stimuli else 'off'] = np.percentile(model_similarities['sim'],
                                                                                         np.arange(0, 100, 5))
            if i == 0:
                models_thresholds['SWOW-RP']['in' if in_stimuli else 'off'] = np.percentile(model_similarities['sim_gt'],
                                                                                                np.arange(0, 100, 5))
                ax_hist[j].hist(model_similarities['sim_gt'], bins=hist_bins, density=True, alpha=0.7, color='blue',
                                label='SWOW-RP', histtype='step')
            ax_hist[j].hist(model_similarities['sim'], bins=hist_bins, density=True, alpha=0.7, color=colors[i],
                            label=model_name, histtype='step')
            ax_hist[j].set_title(f'Similarity distributions for words {"in" if in_stimuli else "off"} stimuli')
            ax_hist[j].set_xlabel('Similarity'), ax_hist[j].set_ylabel('Density')
            ax_hist[j].legend()
    fig_hist.savefig(save_path / 'similarities_hist.png')
    plt.show()

    return models_thresholds
