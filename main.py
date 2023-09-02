import argparse
from corpora import Corpora
from pathlib import Path
from gensim.models import Word2Vec


def train(corpus, file_path):
    model = Word2Vec(sentences=corpus, vector_size=300, window=5, min_count=1, workers=-1)
    model.save(str(file_path))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpora', type=str, default='wikidump+texts',
                        help='Texts to be employed for training')
    parser.add_argument('-d', '--dataset', type=str, default='large_datasets/eswiki-20230820-pages-articles.xml.bz2',
                        help='Path to large text dataset')
    parser.add_argument('-m', '--model', type=str, default='wikis_texts', help='Model name')
    parser.add_argument('-s', '--split', type=float, default=0.1, help='Split for baseline corpus')
    parser.add_argument('-o', '--output', type=str, default='model', help='Where to save the trained model')
    args = parser.parse_args()
    model_path = Path(args.output, args.model)

    corpora = args.corpora.split('+')
    training_corpora = Corpora()
    for corpus in corpora:
        datapath = Path(args.dataset) if corpus == 'wikidump' else Path(corpus)
        training_corpora.add_corpus(corpus, datapath, args.split)
    model_path.parent.mkdir(exist_ok=True)
    train(training_corpora, model_path)
