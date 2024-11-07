import numpy as np
import seaborn as sns
from pandas import DataFrame
from matplotlib import pyplot as plt, colormaps
from numpy import linspace


def plot_correlations(models_results, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for i, stimuli in enumerate(['in_stimuli', 'off_stimuli']):
        title = f'{"in" if stimuli == "in_stimuli" else "off"} stimuli words'
        data = DataFrame(models_results[stimuli])
        sns.boxplot(data=data, ax=ax[i], width=0.5)
        ax[i].set_title(title)
        ax[i].set_xlabel('Model')
        ax[i].set_ylabel('Spearman correlation coefficient')
        plt.setp(ax[i].xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(save_path / 'correlations.png')
    plt.show()


def plot_loss(loss_sg, loss_fix, model_name, save_path, model='W2V'):
    sns.set_theme()
    plt.plot(loss_sg, label=model, alpha=0.7)
    plt.plot(loss_fix, label='Fix duration', alpha=0.7)
    plt.legend()
    plt.xlabel('Step')
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
