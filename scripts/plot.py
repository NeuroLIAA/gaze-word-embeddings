from seaborn import color_palette, set_theme, stripplot, pointplot, scatterplot, set_style, set_context, despine
from matplotlib import pyplot as plt
from numpy import array, percentile
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, MDS
from scripts.utils import save_results
from pandas import DataFrame
import umap

CATEGORIES_COLORS = color_palette('Paired') + ['darkgrey', 'gold', 'cyan']


def print_values(results_df):
    for model in results_df.columns:
        print(f'{model}: {results_df[model].mean():.4f} ± {results_df[model].sem():.4f}')


def plot_results(ax, results_df, label, model_type):
    stripplot(data=results_df, ax=ax, alpha=.3)
    pointplot(data=results_df, linestyles='dotted', color='black', ax=ax, errorbar='sd')
    ax.set_title(f'{model_type.upper()}')
    ax.set_ylabel(label)


def plot_distribution(results_dict, save_path, label, ylabel, fig_title):
    skip_df, lstm_df = save_results(results_dict, save_path, label)
    set_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    plot_results(ax1, skip_df, ylabel, 'w2v')
    plot_results(ax2, lstm_df, ylabel, 'lstm')
    print_values(skip_df), print_values(lstm_df)
    fig.suptitle(fig_title)
    plt.tight_layout()
    plt.savefig(save_path / f'{label}.png', dpi=300)
    plt.show()


def plot_loss(loss_sg, loss_fix, model_name, save_path, model='W2V'):
    set_theme()
    plt.plot(loss_sg, label=model, alpha=0.7)
    plt.plot(loss_fix, label='Fix duration', alpha=0.7)
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'{model_name} loss')
    plt.savefig(save_path / 'loss.png')


def plot_ppl(ppl, model_name, save_path):
    set_theme()
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
    set_style("white")
    set_context("paper")

    if categories_scores is not None and color_by_category and 'category' in df.columns:
        fig = plt.figure(figsize=figsize, facecolor='white')
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.1)
        ax = fig.add_subplot(gs[0])
    else:
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    if color_by_category and 'category' in df.columns:
        unique_categories = sorted(df['category'].unique())
        color_map = {}
        color_idx = 0
        for category in unique_categories:
            if category == 'otro':
                color_map[category] = 'lightgrey'
            else:
                color_map[category] = CATEGORIES_COLORS[color_idx]
                color_idx += 1
        df_otro = df[df['category'] == 'otro']
        df_categorized_words = df[df['category'] != 'otro']
        if len(df_otro) > 0:
            scatterplot(data=df_otro, x='dim1', y='dim2',
                            hue='category', palette={'otro': 'lightgrey'},
                            s=20, alpha=0.3, edgecolor='white', linewidth=0.5,
                            legend=True, ax=ax)
        if len(df_categorized_words) > 0:
            scatterplot(data=df_categorized_words, x='dim1', y='dim2',
                            hue='category', palette=color_map,
                            s=35, alpha=0.8, edgecolor='grey', linewidth=0.3,
                            legend=True, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels,
                  loc='best', frameon=True, fancybox=True,
                  shadow=True, fontsize=10)
    else:
        scatterplot(data=df, x='dim1', y='dim2',
                        color='steelblue', s=30, alpha=0.7,
                        edgecolor='white', linewidth=0.5, ax=ax)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(f'{model_name} word embeddings', fontsize=12, pad=20)
    despine(ax=ax)
    if save_path:
        fig.savefig(save_path / f'umap_{model_name}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_semantic_scores(semantic_clustering_dict, models_dirs, save_path):
    semantic_clustering_df = DataFrame.from_dict(semantic_clustering_dict, orient='index')
    for model in models_dirs:
        model_name = model.name
        model_scores = [column for column in semantic_clustering_df.columns if model_name in column]
        model_df = semantic_clustering_df[model_scores]
        model_df.columns = [col.replace(f'{model_name}_', '') for col in model_df.columns]
        model_df.to_csv(save_path / f'semantic_clustering_{model_name}.csv')
        plot_model_semantic_scores(model_df, save_path, f'semantic_clustering_{model_name}.png')


def plot_model_semantic_scores(semantic_clustering_df, save_path, filename):
    labels = semantic_clustering_df.index.tolist()
    models = semantic_clustering_df.columns.tolist()
    n_labels, n_models = len(labels), len(models)
    gap = 1
    bar_width = 1.0

    xs, vals, labs, mods = [], [], [], []
    for i, label in enumerate(labels):
        base = i * (n_models + gap)
        for j, model in enumerate(models):
            xs.append(base + j)
            vals.append(semantic_clustering_df.loc[label, model])
            labs.append(label)
            mods.append(model)

    total_span = max(xs) - min(xs) + 1
    fig_width = max(10, min(28, total_span * 0.22))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    for i, label in enumerate(labels):
        idxs = [k for k, l in enumerate(labs) if l == label]
        xpos = [xs[k] for k in idxs]
        yvals = [vals[k] for k in idxs]
        color = CATEGORIES_COLORS[i % len(CATEGORIES_COLORS)]
        ax.bar(xpos, yvals, width=bar_width, color=color, label=label, edgecolor='k', linewidth=0.4, alpha=0.75)

    ax.set_xticks(xs)
    ax.set_xticklabels(mods, rotation=90, fontsize=8)
    for i in range(1, n_labels):
        sep_x = i * (n_models + gap) - 1.0
        ax.axvline(sep_x, color='k', linestyle='--', linewidth=0.4, alpha=0.35)

    ax.set_xlabel('Models')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Semantic clustering values per label across models')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small')
    left_xlim = min(xs) - (bar_width / 1.2)
    right_xlim = max(xs) + (bar_width / 1.2)
    ax.set_xlim(left_xlim, right_xlim)
    plt.tight_layout()
    plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
    plt.show()
