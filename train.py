import logging
import argparse
from scripts.corpora import Corpora, load_corpora
from scripts.utils import get_embeddings_path
from models.word2vec_gensim.model import Word2vecGensim
from models.word2vec_torch.model import Word2vecTorch
from models.lstm.main import AwdLSTM


class Trainer:
    def __init__(self, corpora_labels, data_sources, fraction, repeats, negative_samples, downsample_factor, epochs, lr, batch_size,
                 device, min_token_len, max_token_len, min_sentence_len, vector_size, window_size, min_count,
                 mode, cbow, train_fix, save_path, tokenizer, max_vocab):
        self.corpora_labels = corpora_labels
        self.data_sources = data_sources
        self.fraction = fraction
        self.repeats = repeats
        self.negative_samples = negative_samples
        self.downsample_factor = downsample_factor
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.min_token_len = min_token_len
        self.max_token_len = max_token_len
        self.min_sentence_len = min_sentence_len
        self.vector_size = vector_size
        self.window_size = window_size
        self.min_count = min_count
        self.mode = mode
        self.cbow = cbow
        self.train_fix = train_fix
        self.save_path = save_path
        self.tokenizer = tokenizer
        self.max_vocab = max_vocab

    def train(self):
        print(f'Beginning training with corpora {self.corpora_labels} ({int(self.fraction * 100)}% of baseline corpus)')
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        
        corpora = load_corpora(self.corpora_labels, self.data_sources, self.fraction, self.repeats, self.min_token_len, 
                               self.max_token_len, self.min_sentence_len, self.tokenizer)
        corpora.print_size()
        
        model_name, save_path = self.get_path(self.save_path, self.corpora_labels, self.data_sources)
        
        model = self.get_model(corpora, model_name, save_path)
        model.train()
        
        print(f'Training completed. Model saved at {save_path}')

    def get_path(self, save_path, corpora_labels, data_sources):
        model_name = corpora_labels[-1] if 'local' in data_sources else 'baseline'
        save_path = save_path / model_name
        return model_name, save_path

    def get_model(self, corpora, model_name, save_path):
        if self.mode == 'word2vec_gensim':
            return Word2vecGensim(corpora, self.vector_size, self.window_size, self.min_count, self.negative_samples,
                                  self.epochs, self.cbow, model_name, save_path)
        elif self.mode == 'word2vec_torch':
            return Word2vecTorch(corpora, self.vector_size, self.window_size, self.min_count, self.negative_samples,
                                 self.downsample_factor, self.epochs, self.lr,
                                 self.batch_size, self.train_fix, self.device, model_name, save_path)
        elif self.mode == 'lstm':
            return AwdLSTM(corpora, model_name, save_path, embed_size=self.vector_size, batch_size=self.batch_size,
                           epochs=self.epochs, lr=self.lr, min_word_count=self.min_count, max_vocab_size=self.max_vocab)
        else:
            raise ValueError(f'Invalid model type: {self.mode}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Model base name')
    parser.add_argument('-c', '--corpora', type=str, default='all_wikis+scanpaths',
                        help='Texts to be employed for training')
    parser.add_argument('-s', '--sources', type=str, default='remote+local',
                        help='Corpora data sources. If remote, will fetch from huggingface\'s large_spanish_corpus')
    parser.add_argument('-f', '--fraction', type=float, default=0.3,
                        help='Fraction of baseline corpus to employ for training')
    parser.add_argument('-cbow', '--cbow', action='store_true', help='If using Gensim, use CBOW instead of SkipGram')
    parser.add_argument('-r', '--repeats', type=int, default=200,
                        help='Number of times the local corpus will be iterated over for training')
    parser.add_argument('-ns', '--negative_samples', type=int, default=20,
                        help='Number of negative samples to be used in training')
    parser.add_argument('-ds', '--downsample_factor', type=float, default=1e-3,
                        help='Downsample factor for frequent words')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('-lr', '--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Device to be used for training (cpu or cuda)')
    parser.add_argument('-min', '--min_count', type=int, default=20, help='Minimum number of occurrences for a word')
    parser.add_argument('-size', '--size', type=int, default=300, help='Size of the word vectors')
    parser.add_argument('-w', '--window', type=int, default=5, help='Window size')
    parser.add_argument('-min_token', '--min_token', type=int, default=2,
                        help='Word min length, in tokens')
    parser.add_argument('-max_token', '--max_token', type=int, default=20,
                        help='Word max length, in tokens')
    parser.add_argument('-min_length', '--min_length', type=int, default=4,
                        help='Sentence minimum length, in tokens')
    parser.add_argument('-tf', '--train_fix', type=str, default='input',
                        help='Train fixation duration regressor of input or output words. Options: input, output.')
    parser.add_argument('-m', '--mode', choices=["word2vec_gensim", "word2vec_torch", "lstm"], type=str, help='Model to be used')
    parser.add_argument('-o', '--output', type=str, default='embeddings', help='Where to save the trained embeddings')
    parser.add_argument('-t', '--tokenizer', action='store_true', help='Use Spacy tokenizer for preprocessing')
    parser.add_argument('-max_vocab', '--max_vocab', type=int, default=None, help='Maximum vocabulary size')
    args = parser.parse_args()
    source_labels, corpora_labels = args.sources.split('+'), args.corpora.split('+')
    if len(source_labels) != len(corpora_labels):
        raise ValueError('You must specify from where each corpus will be fetched')
    model_path = get_embeddings_path(args.output, args.model, args.fraction)
    
    #pepe3 -c "all_wikis" -s "remote" -f 0.01 -m "lstm" -lr 30 -t -max_vocab 30000 -min 5

    Trainer(corpora_labels, source_labels, args.fraction, args.repeats, args.negative_samples, args.downsample_factor,
            args.epochs, args.lr, args.batch_size, args.device, args.min_token, args.max_token, args.min_length,
            args.size, args.window, args.min_count, args.mode, args.cbow, args.train_fix, model_path, args.tokenizer,
            args.max_vocab).train()
