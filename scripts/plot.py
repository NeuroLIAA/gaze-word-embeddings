import seaborn as sns
from matplotlib import pyplot as plt
from numpy import array, percentile
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, MDS
from scripts.utils import save_results
import umap


def print_values(results_df):
    for model in results_df.columns:
        print(f'{model}: {results_df[model].mean():.4f} ± {results_df[model].sem():.4f}')


def plot_results(ax, results_df, label, model_type):
    sns.stripplot(data=results_df, ax=ax, alpha=.3)
    sns.pointplot(data=results_df, linestyles='dotted', color='black', ax=ax, errorbar='sd')
    ax.set_title(f'{model_type.upper()}')
    ax.set_ylabel(label)


def plot_distribution(results_dict, save_path, label, ylabel, fig_title):
    skip_df, lstm_df = save_results(results_dict, save_path, label)
    sns.set_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    plot_results(ax1, skip_df, ylabel, 'w2v')
    plot_results(ax2, lstm_df, ylabel, 'lstm')
    print_values(skip_df), print_values(lstm_df)
    fig.suptitle(fig_title)
    plt.tight_layout()
    plt.savefig(save_path / f'{label}.png', dpi=300)
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
    sns.set_theme()
    plt.figure(figsize=(10, 5))
    plt.plot(ppl, label=model_name, alpha=0.7, marker='o')
    plt.title(model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.savefig(save_path / 'ppl.png')


def reduce_dimensionality(embeddings, method='pca', n_components=2, **kwargs):
    embeddings = array(embeddings)

    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components, **kwargs)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, **kwargs)
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42, n_jobs=1, **kwargs)
    elif method.lower() == 'isomap':
        reducer = Isomap(n_components=n_components, **kwargs)
    elif method.lower() == 'mds':
        reducer = MDS(n_components=n_components, random_state=42, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'pca', 'tsne', 'umap', 'isomap', 'mds'")

    return reducer.fit_transform(embeddings)


def remove_outliers(embeddings, words, method='iqr', threshold=1.5):
    embeddings = array(embeddings)

    if method == 'iqr':
        mask = [True] * len(embeddings)
        for dim in range(embeddings.shape[1]):
            Q1 = percentile(embeddings[:, dim], 25)
            Q3 = percentile(embeddings[:, dim], 75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            mask = mask & (embeddings[:, dim] >= lower) & (embeddings[:, dim] <= upper)
    elif method == 'percentile':
        mask = [True] * len(embeddings)
        for dim in range(embeddings.shape[1]):
            lower = percentile(embeddings[:, dim], (100 - threshold) / 2)
            upper = percentile(embeddings[:, dim], 100 - (100 - threshold) / 2)
            mask = mask & (embeddings[:, dim] >= lower) & (embeddings[:, dim] <= upper)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'iqr' or 'percentile'")

    clean_embeddings = embeddings[mask]
    clean_words = [w for w, m in zip(words, mask) if m]

    return clean_embeddings, clean_words


def plot_embeddings(model_wv, words_data, model_name,
                    remove_outliers_flag=True, outlier_method='iqr',
                    outlier_threshold=1.5, figsize=(12, 8), color_by_category=True,
                    save_path=None, dpi=300, categories_scores=None):
    df = words_data.copy()
    if color_by_category and 'category' not in df.columns:
        raise ValueError("DataFrame must contain a 'category' column when color_by_category=True")
    words = df.index.tolist()
    embeddings = array(model_wv[words])
    embeddings = reduce_dimensionality(embeddings, method='umap')
    if remove_outliers_flag:
        embeddings_clean, words_clean = remove_outliers(embeddings, words, method=outlier_method,
                                                        threshold=outlier_threshold)
        df = df.loc[words_clean]
        embeddings = embeddings_clean
        print(f"Removed {len(words) - len(words_clean)} outliers")
    df['dim1'] = embeddings[:, 0]
    df['dim2'] = embeddings[:, 1]
    sns.set_style("white")
    sns.set_context("paper")

    if categories_scores is not None and color_by_category and 'category' in df.columns:
        fig = plt.figure(figsize=figsize, facecolor='white')
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.1)
        ax = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax_bar = None

    if color_by_category and 'category' in df.columns:
        unique_categories = sorted(df['category'].unique())
        n_colors = len([c for c in unique_categories if c != 'otro'])
        if n_colors > 12:
            extra_colors = ['darkgrey', 'gold', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink']
            colors = sns.color_palette('Paired') + extra_colors
        else:
            colors = sns.color_palette('Set1', n_colors=n_colors)
        color_map = {}
        color_idx = 0
        for category in unique_categories:
            if category == 'otro':
                color_map[category] = 'lightgrey'
            else:
                color_map[category] = colors[color_idx]
                color_idx += 1
        df_otro = df[df['category'] == 'otro']
        df_categorized_words = df[df['category'] != 'otro']
        if len(df_otro) > 0:
            sns.scatterplot(data=df_otro, x='dim1', y='dim2',
                            hue='category', palette={'otro': 'lightgrey'},
                            s=20, alpha=0.3, edgecolor='white', linewidth=0.5,
                            legend=True, ax=ax)
        if len(df_categorized_words) > 0:
            sns.scatterplot(data=df_categorized_words, x='dim1', y='dim2',
                            hue='category', palette=color_map,
                            s=35, alpha=0.8, edgecolor='grey', linewidth=0.3,
                            legend=True, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels,
                  loc='best', frameon=True, fancybox=True,
                  shadow=True, fontsize=10)

        if categories_scores is not None and ax_bar is not None:
            filtered_scores = {k: v for k, v in categories_scores.items() if k != 'otro'}
            if filtered_scores:
                sorted_categories = [cat for cat in sorted(unique_categories) if cat in filtered_scores]
                scores = [filtered_scores[cat] for cat in sorted_categories]
                bar_colors = [color_map[cat] for cat in sorted_categories]
                _ = ax_bar.bar(range(len(sorted_categories)), scores, color=bar_colors,
                                  edgecolor='black', linewidth=0.5, alpha=0.8)
                ax_bar.set_xticks(range(len(sorted_categories)))
                ax_bar.set_xticklabels(sorted_categories, rotation=45, ha='right', fontsize=9)
                ax_bar.set_ylabel('Silhouette Score', fontsize=9)
                ax_bar.set_ylim((-0.1, 0.6))
                ax_bar.grid(axis='y', alpha=0.8, linestyle='--', linewidth=0.5)
                sns.despine(ax=ax_bar)

    else:
        sns.scatterplot(data=df, x='dim1', y='dim2',
                        color='steelblue', s=30, alpha=0.7,
                        edgecolor='white', linewidth=0.5, ax=ax)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(f'{model_name} word embeddings', fontsize=12, pad=20)
    sns.despine(ax=ax)
    if save_path:
        fig.savefig(save_path / f'umap_{model_name}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
