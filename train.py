import logging
import argparse
from pathlib import Path
from scripts.corpora import load_corpora, load_gaze_table
from scripts.utils import get_embeddings_path
from models.word2vec.model import W2VTrainer
from models.lstm.main import AwdLSTM
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class Trainer:
    def __init__(self, corpora_labels, data_sources, name, fraction, repeats, negative_samples, downsample_factor,
                 epochs, lr, min_lr, batch_size, device, min_token_len, max_token_len, min_sentence_len,
                 max_sentence_len, vector_size, window_size, min_count, model, gaze_table, save_path, pretrained_path,
                 tokenizer, max_vocab, stimuli_path, pretrained_embeddings_path):
        self.corpora_labels = corpora_labels
        self.data_sources = data_sources
        self.name = name
        self.fraction = fraction
        self.repeats = repeats
        self.negative_samples = negative_samples
        self.downsample_factor = downsample_factor
        self.epochs = epochs
        self.lr = lr
        self.min_lr = min_lr
        self.batch_size = batch_size
        self.device = device
        self.min_token_len = min_token_len
        self.max_token_len = max_token_len
        self.min_sentence_len = min_sentence_len
        self.max_sentence_len = max_sentence_len
        self.vector_size = vector_size
        self.window_size = window_size
        self.min_count = min_count
        self.model = model
        self.gaze_table = gaze_table
        self.save_path = save_path
        self.pretrained_path = pretrained_path
        self.tokenizer = tokenizer
        self.max_vocab = max_vocab
        self.stimuli_path = stimuli_path
        self.pretrained_embeddings_path = pretrained_embeddings_path

    def train(self):
        print(f'Beginning training with corpora {self.corpora_labels} ({int(self.fraction * 100)}% of baseline corpus)')
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        corpora = load_corpora(self.corpora_labels, self.data_sources, self.fraction, self.repeats, self.min_token_len, 
                               self.max_token_len, self.min_sentence_len, self.max_sentence_len, self.tokenizer)
        corpora.print_size()
        model = self.get_model(corpora)
        model.train()
        print(f'Training completed. Model saved at {self.save_path}')

    def get_model(self, corpora):
        model_name = self.set_paths()
        if self.model == 'skip' or self.model == 'cbow':
            return W2VTrainer(corpora, self.vector_size, self.window_size, self.min_count, self.negative_samples,
                              self.downsample_factor, self.epochs, self.lr, self.min_lr, self.batch_size,
                              self.gaze_table, self.stimuli_path, self.device, model_name, self.model,
                              self.pretrained_path, self.save_path)
        elif self.model == 'lstm':
            return AwdLSTM.create_from_args(corpora, model_name, self.save_path, self.pretrained_path, self.stimuli_path,
                                            embed_size=self.vector_size, batch_size=self.batch_size, epochs=self.epochs,
                                            lr=self.lr, min_word_count=self.min_count, max_vocab_size=self.max_vocab,
                                            pretrained_embeddings_path=self.pretrained_embeddings_path)
        else:
            raise ValueError(f'Invalid model type: {self.model}')

    def set_paths(self):
        corpus_name = self.corpora_labels[-1] if 'local' in self.data_sources else 'baseline'
        model_name = f'{self.model}_{corpus_name}'
        if self.name:
            model_name += f'_{self.name}'
        self.pretrained_path = self.save_path / self.pretrained_path if self.pretrained_path else None
        self.save_path = self.save_path / model_name
        return model_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='Training data descriptive name')
    parser.add_argument('-n', '--name', type=str, default='', help='(Optional) suffix name for the model')
    parser.add_argument('-c', '--corpora', type=str, default='all_wikis',
                        help='Texts to be employed for training')
    parser.add_argument('-s', '--sources', type=str, default='remote',
                        help='Corpora data sources. If remote, will fetch from huggingface\'s large_spanish_corpus')
    parser.add_argument('-f', '--fraction', type=float, default=1.0,
                        help='Fraction of baseline corpus to employ for training')
    parser.add_argument('-r', '--repeats', type=int, default=1,
                        help='Number of times the local corpus will be iterated over for training')
    parser.add_argument('-ns', '--negative_samples', type=int, default=20,
                        help='Number of negative samples to be used in training for Word2Vec')
    parser.add_argument('-ds', '--downsample_factor', type=float, default=1e-5,
                        help='Downsample factor for frequent words in Word2Vec')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('-lr', '--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('-min_lr', '--min_lr', type=float, default=1e-4, help='Minimum learning rate')
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
    parser.add_argument('-max_length', '--max_length', type=int, default=40,
                        help='Sentence maximum length, in tokens')
    parser.add_argument('-t', '--tokenizer', action='store_true', help='Use Spacy tokenizer for preprocessing')
    parser.add_argument('-max_vocab', '--max_vocab', type=int, default=None, help='Maximum vocabulary size')
    parser.add_argument('-st', '--stimuli', type=str, default='stimuli',
                        help='Path to text files employed in the experiment')
    parser.add_argument('-gt', '--gaze_table', type=str, default='words_measurements.pkl',
                        help='Path to gaze measurements table')
    parser.add_argument('-gf', '--gaze_features', type=str, nargs='*', default=[],
                        help='Gaze features to be employed in training')
    parser.add_argument('-m', '--model', choices=['skip', 'cbow', 'lstm'], type=str,
                        help='Model architecture to be trained')
    parser.add_argument('-ft', '--finetune', type=str, default=None,
                        help='Path to pre-trained model to be fine-tuned')
    parser.add_argument('-pte', '--pretrained_embeddings', type=str, help='Path to pre-trained embeddings')
    parser.add_argument('-o', '--output', type=str, default='embeddings', help='Where to save the trained embeddings')
    args = parser.parse_args()
    source_labels, corpora_labels = args.sources.split('+'), args.corpora.split('+')
    if len(source_labels) != len(corpora_labels):
        raise ValueError('You must specify from where each corpus will be fetched')
    stimuli_path, gaze_table_path = Path(args.stimuli), Path(args.gaze_table)
    if not stimuli_path.exists():
        raise FileNotFoundError(f'Stimuli path {args.stimuli} does not exist')
    gaze_table = load_gaze_table(gaze_table_path, args.gaze_features)
    save_path = get_embeddings_path(args.output, args.data, args.fraction)
    
    #test -c "all_wikis" -s "remote" -f 0.01 -m "lstm" -lr 30 -t -e 5 -st "./stimuli"
    #test -c "all_wikis" -s "remote" -f 0.01 -m "lstm" -lr 30 -t -e 5 -st "./stimuli" -pte "./embeddings/all_wikis/w2v_baseline"
    #test -c "scanpaths" -s "local" -f 1 -m "lstm" -lr 30 -t -e 5 -st "./stimuli" -ft "lstm_baseline"

    Trainer(corpora_labels, source_labels, args.name, args.fraction, args.repeats, args.negative_samples,
            args.downsample_factor, args.epochs, args.lr, args.min_lr, args.batch_size, args.device, args.min_token,
            args.max_token, args.min_length, args.max_length, args.size, args.window, args.min_count, args.model,
            gaze_table, save_path, args.finetune, args.tokenizer, args.max_vocab, stimuli_path,
            args.pretrained_embeddings).train()
