import argparse
from pathlib import Path
from gensim.models import Word2Vec
from datasets import load_dataset


def train(corpus, model_path):
    model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=5, workers=4)
    model.save(str(model_path))


def load_baseline_corpus(dataset='large_spanish_corpus', name='all_wiki', split='10%'):
    baseline_corpus = load_dataset(dataset, name, split=f'train[:{split}]')

    return baseline_corpus


def load_corpus(path):
    files = [f for f in path.iterdir()]
    corpus = []
    for file in files:
        if file.is_file():
            with file.open('r') as f:
                corpus.append(f.read())
        elif file.is_dir():
            corpus += load_corpus(file)

    return corpus


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', type=str, default='corpus', help='Path to training corpus')
    parser.add_argument('-m', '--model', type=str, default='w2v', help='Model name')
    parser.add_argument('-o', '--output', type=str, default='model', help='Where to save trained model')
    parser.add_argument('-d', '--dataset', type=str, default='all_wiki',
                        help='Dataset name of baseline corpus')
    parser.add_argument('-s', '--split', type=str, default='10%', help='Split for baseline corpus')
    args = parser.parse_args()

    corpus_path, model_path = Path(args.corpus), Path(args.output, args.model)
    corpus = load_corpus(corpus_path)
    train(corpus, model_path)
