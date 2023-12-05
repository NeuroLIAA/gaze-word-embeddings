import logging
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
from scripts.corpora import Corpora, get_dataloader_and_vocab
from scripts.utils import get_model_path
from scripts.w2v import SkipGram
from gensim.models import Word2Vec


def train(corpora_labels, data_sources, fraction, repeats, negative_samples, downsample_factor, epochs, lr, batch_size,
          device, min_token_len, max_token_len, min_sentence_len, vector_size, window_size, min_count,
          gensim, cbow, save_path):
    print(f'Beginning training with corpora {corpora_labels} ({int(fraction * 100)}% of baseline corpus)')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    corpora = load_corpora(corpora_labels, data_sources, fraction, repeats,
                           min_token_len, max_token_len, min_sentence_len)
    corpora.print_size()
    model_name, save_path = get_path(save_path, corpora_labels, data_sources)
    if gensim:
        train_gensim(corpora, vector_size, window_size, min_count, negative_samples,
                     epochs, cbow, model_name, save_path)
    else:
        train_torch(corpora, vector_size, window_size, min_count, negative_samples, downsample_factor, epochs,
                    lr, batch_size, device, model_name, save_path)
    print(f'Training completed. Model saved at {save_path}')


def train_gensim(corpora, vector_size, window_size, min_count, negative_samples, epochs, cbow, model_name, save_path):
    model = Word2Vec(sentences=corpora, sg=not cbow, vector_size=vector_size, window=window_size, min_count=min_count,
                     negative=negative_samples, epochs=epochs, workers=12)
    save_path.mkdir(exist_ok=True, parents=True)
    model.wv.save_word2vec_format(str(save_path / f'{model_name}.vec'))


def train_torch(corpora, vector_size, window_size, min_count, negative_samples, downsample_factor, epochs, lr,
                batch_size, device, model_name, save_path):
    dataloader, vocab = get_dataloader_and_vocab(corpora, min_count, negative_samples, downsample_factor,
                                                 window_size, batch_size)
    skip_gram = SkipGram(len(vocab), vector_size)
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        skip_gram.cuda()
    else:
        device = torch.device('cpu')

    for epoch in range(epochs):
        print(f'\nEpoch: {epoch + 1}')
        optimizer = optim.SparseAdam(skip_gram.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))
        for batch in tqdm(dataloader):
            if len(batch[0]) > 1:
                pos_u = batch[0].to(device)
                pos_v = batch[1].to(device)
                neg_v = batch[2].to(device)

                optimizer.zero_grad()
                loss = skip_gram.forward(pos_u, pos_v, neg_v)
                loss.backward()
                optimizer.step()
                scheduler.step()

    save_path.mkdir(exist_ok=True, parents=True)
    skip_gram.save_embedding_vocab(vocab, str(save_path / f'{model_name}.vec'))


def load_corpora(corpora_labels, data_sources, fraction, repeats, min_token_len, max_token_len, min_sentence_len):
    training_corpora = Corpora(min_token_len, max_token_len, min_sentence_len)
    for corpus, source in zip(corpora_labels, data_sources):
        training_corpora.add_corpus(corpus, source, fraction, repeats)
    return training_corpora


def get_path(save_path, corpora_labels, data_sources):
    model_name = corpora_labels[-1] if 'local' in data_sources else 'baseline'
    save_path = save_path / model_name
    return model_name, save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Model base name')
    parser.add_argument('-c', '--corpora', type=str, default='all_wikis+scanpaths',
                        help='Texts to be employed for training')
    parser.add_argument('-s', '--sources', type=str, default='remote+local',
                        help='Corpora data sources. If remote, will fetch from huggingface\'s large_spanish_corpus')
    parser.add_argument('-f', '--fraction', type=float, default=1.0,
                        help='Fraction of baseline corpus to employ for training')
    parser.add_argument('-cbow', '--cbow', action='store_true', help='If using Gensim, use CBOW instead of SkipGram')
    parser.add_argument('-r', '--repeats', type=int, default=1,
                        help='Number of times the local corpus will be iterated over for training')
    parser.add_argument('-ns', '--negative_samples', type=int, default=20,
                        help='Number of negative samples to be used in training')
    parser.add_argument('-ds', '--downsample_factor', type=float, default=1e-3,
                        help='Downsample factor for frequent words')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('-lr', '--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='Device to be used for training (cpu or cuda)')
    parser.add_argument('-min', '--min_count', type=int, default=20, help='Minimum number of occurrences for a word')
    parser.add_argument('-size', '--size', type=int, default=300, help='Size of the word vectors')
    parser.add_argument('-w', '--window', type=int, default=5, help='Window size')
    parser.add_argument('-min_token', '--min_token', type=int, default=2,
                        help='Word min length, in tokens')
    parser.add_argument('-max_token', '--max_token', type=int, default=20,
                        help='Word max length, in tokens')
    parser.add_argument('-min_length', '--min_length', type=int, default=10,
                        help='Sentence min length, in tokens, for large scale corpora')
    parser.add_argument('-g', '--gensim', action='store_true', help='Use gensim instead of PyTorch')
    parser.add_argument('-o', '--output', type=str, default='models', help='Where to save the trained models')
    args = parser.parse_args()
    source_labels, corpora_labels = args.sources.split('+'), args.corpora.split('+')
    if len(source_labels) != len(corpora_labels):
        raise ValueError('You must specify from where each corpus will be fetched')
    model_path = get_model_path(args.output, args.model, args.fraction)

    train(corpora_labels, source_labels, args.fraction, args.repeats, args.negative_samples, args.downsample_factor,
          args.epochs, args.lr, args.batch_size, args.device, args.min_token, args.max_token, args.min_length,
          args.size, args.window, args.min_count, args.gensim, args.cbow, model_path)
