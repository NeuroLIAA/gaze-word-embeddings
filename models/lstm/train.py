from models.lstm.main import AwdLSTM
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
import timeit
import warnings
import random

from tqdm import tqdm

from models.lstm.model import Model
from models.lstm.ntasgd import NTASGD

from scripts.data_handling import get_vocab
from scripts.utils import get_words_in_corpus
from scripts.plot import plot_loss

from gensim.models import KeyedVectors
from pandas import read_csv

class AwdLSTMForTraining(AwdLSTM):
    def __init__(self, corpora, name, save_path, pretrained_model_path, stimuli_path, layer_num, embed_size, hidden_size, 
                 lstm_type, w_drop, dropout_i, dropout_l, dropout_o, dropout_e, winit, batch_size, 
                 valid_batch_size, bptt, ar, tar, weight_decay, epochs, lr, max_grad_norm, non_mono, 
                 device, log, min_word_count, max_vocab_size, shard_count, pretrained_embeddings_path):
        super().__init__(corpora, name, save_path, pretrained_model_path, stimuli_path, layer_num, embed_size, hidden_size, 
                         lstm_type, w_drop, dropout_i, dropout_l, dropout_o, dropout_e, winit, batch_size, valid_batch_size, 
                         bptt, ar, tar, weight_decay, epochs, lr, max_grad_norm, non_mono, device, log, min_word_count, max_vocab_size, 
                         shard_count, pretrained_embeddings_path)
        self.data_init()

    def data_init(self):
        text = self.corpora
        split = text.corpora.train_test_split(test_size=0.2, seed=self.SEED)

        data = split['train']
        vld_data = split['test']

        vocab = self.generate_vocab(data, self.save_path / 'vocab.pt')

        print("Numericalizing Training set")
        data = data.map(
            lambda row: {"text": vocab(row["text"]), "fix_dur": row["fix_dur"]}, num_proc=12
        )
        
        print("Numericalizing Validation set")
        vld_data = vld_data.map(
            lambda row: {"text": vocab(row["text"]), "fix_dur": row["fix_dur"]}, num_proc=12
        )

        print("Reshaping Training set")
        data = data.map(self.chunk_examples, batched=True, remove_columns=data.column_names)

        print("Reshaping Validation set")
        vld_data = vld_data.map(self.chunk_examples, batched=True, remove_columns=vld_data.column_names)
        self.vocab = vocab.get_stoi()
        
        data = data.with_format("torch")
        vld_data = vld_data.with_format("torch")

        self.data = data
        print("Loading validation tokens...")
        self.vld_data_tokens = vld_data["text"].reshape(-1, 1)
        print("Loading validation fix durations...")
        self.vld_data_fix = vld_data["fix_dur"].reshape(-1, 1)

    def generate_vocab(self, data, vocab_savepath=None):
        if self.pretrained_embeddings_path is not None:
            print("Loading embeddings pretrained vocab...")
            vocab, _, _, _ = torch.load(self.pretrained_embeddings_path / "vocab.pt", weights_only=False).values()
            return vocab
        else:
            words_in_stimuli = get_words_in_corpus(self.stimuli_path)
            return get_vocab(corpora=data, 
                            min_count=self.min_word_count, 
                            words_in_stimuli=words_in_stimuli, 
                            max_vocab_size=self.max_vocab_size, 
                            is_baseline=self.pretrained_model_path is None,
                            vocab_savepath=vocab_savepath)[0]

    def compare_embeddings(self, model, epoch):
        print("Generating embeddings...")
        self.generate_embeddings(model)

        print("Loading embeddings...")
        embeddings = KeyedVectors.load_word2vec_format(str(self.save_path / f'{self.name}.vec'), binary=False)

        simlex = read_csv('evaluation/spa.csv')
        simlex['word1'] = simlex['word1'].str.lower()
        simlex['word2'] = simlex['word2'].str.lower()

        simlex['similarity'] = simlex.apply(lambda row: embeddings.similarity(row['word1'], row['word2']) if (row['word1'] in embeddings and row['word2'] in embeddings) else np.nan, axis=1)
        simlex.dropna(inplace=True)

        corr = spearmanr(simlex['similarity'], simlex['score'])

        self.simlex_file.write("{:d},{:.4f},{:.4f}\n".format(epoch, corr.correlation, corr.pvalue))

        print(f'Simlex correlation: {corr.correlation:.4f}')
        print(f'Simlex p-value: {corr.pvalue:.4f}')
        return corr.correlation, corr.pvalue

    def train_model(self, model, optimizer):
        tic = timeit.default_timer()
        print("Starting training.")
        best_val = 1e10
        metrics = { "loss_sg": [], "loss_fix": [] }
        for epoch in range(self.epochs):
            self.train_epoch(model, optimizer, epoch + 1, metrics)

            tmp = {}
            for (prm, st) in optimizer.state.items():
                tmp[prm] = prm.clone().detach()
                prm.data = st['ax'].clone().detach()

            val_perp = self.perplexity(self.vld_data_tokens, self.vld_data_fix, model)
            optimizer.check(val_perp)

            self.log_file.write("{:d},{:.3f}\n".format(epoch + 1, val_perp))

            if val_perp < best_val:
                best_val = val_perp
                print("Best validation perplexity : {:.3f}".format(best_val))
                self.save_model(model)
                print("Model saved!")

            for (prm, st) in optimizer.state.items():
                prm.data = tmp[prm].clone().detach()

            self.compare_embeddings(model, epoch)

            toc = timeit.default_timer()
            print("Validation set perplexity : {:.3f}".format(val_perp))
            print("Since beginning : {:.3f} mins".format(round((toc - tic) / 60)))
            print("*************************************************\n")
        self.log_file.close()
        self.plot_loss(metrics['loss_sg'], metrics['loss_fix'])
        self.generate_embeddings(model)
    
    def load_pretrained_embeddings(self, model):
        if self.pretrained_embeddings_path is not None:
            print("Loading pretrained embeddings...")
            pretrained_embeddings_state = torch.load(
                self.pretrained_embeddings_path / f"{self.pretrained_embeddings_path.name}.pt",
                map_location=torch.device("cpu")
            )
            pretrained_embeddings = pretrained_embeddings_state['model_state_dict']['u_embeddings.weight']
            
            model_state = model.state_dict()
            model_state['embed.W'] = pretrained_embeddings

            model.load_state_dict(model_state) 
        
    def train(self):
        warnings.filterwarnings("ignore")
        self.vld_data_tokens, self.vld_data_fix = self.batchify(self.vld_data_tokens, self.vld_data_fix, self.valid_batch_size)
        model = Model(len(self.vocab), self.embed_size, self.hidden_size, self.layer_num, self.w_drop, self.dropout_i,
                      self.dropout_l, self.dropout_o, self.dropout_e, self.winit, self.lstm_type)
        model.to(self.device)
        self.load_pretrained_embeddings(model)
        optimizer = NTASGD(model.parameters(), lr=self.lr, n=self.non_mono, weight_decay=self.weight_decay, fine_tuning=False)
        self.train_model(model, optimizer)
        