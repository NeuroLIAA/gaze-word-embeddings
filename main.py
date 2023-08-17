import argparse
from pathlib import Path


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
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    corpus = load_corpus(corpus_path)
    print(corpus)
