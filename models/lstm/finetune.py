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

class AwdLSTMForFinetuning(AwdLSTM):
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
        data = self.corpora.corpora

        vocab = self.generate_vocab(data, self.pretrained_model_path / 'vocab.pt')

        print("Numericalizing Training set")
        data = data.map(
            lambda row: {"text": vocab(row["text"]), "fix_dur": row["fix_dur"]}, num_proc=12
        )

        print("Reshaping Training set")
        data = data.map(self.chunk_examples, batched=True, remove_columns=data.column_names, num_proc=12)

        self.vocab = vocab.get_stoi()
        
        data = data.with_format("torch")

        self.data = data

    def train_model(self, model, optimizer):
        tic = timeit.default_timer()
        print("Starting finetuning.")
        metrics = { "loss_sg": [], "loss_fix": [], "fix_corrs": [], "fix_pvalues": [] }
        for epoch in range(self.epochs):
            self.train_epoch(model, optimizer, epoch + 1, metrics)

            self.save_model(model)
            toc = timeit.default_timer()
            print("Since beginning : {:.3f} mins".format(round((toc - tic) / 60)))
            print(f'Fix duration correlation: {np.nanmean(metrics['fix_corrs']):.4f} (+/- {np.nanstd(metrics['fix_corrs']):.4f})')
            print(f'Fix duration p-value: {np.nanmean(metrics['fix_pvalues']):.4f} (+/- {np.nanstd(metrics['fix_pvalues']):.4f})')
            print("*************************************************\n")
        self.log_file.close()
        self.plot_loss(metrics['loss_sg'], metrics['loss_fix'])
        self.generate_embeddings(model)
        
    def train(self):
        warnings.filterwarnings("ignore")
        model = Model(len(self.vocab), self.embed_size, self.hidden_size, self.layer_num, self.w_drop, self.dropout_i,
                      self.dropout_l, self.dropout_o, self.dropout_e, self.winit, self.lstm_type)
        model.to(self.device)
        checkpoint = torch.load(self.checkpoint())
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = NTASGD(model.parameters(), lr=self.lr, n=self.non_mono, weight_decay=self.weight_decay, fine_tuning=True)
        self.train_model(model, optimizer)

    def checkpoint(self):
        return next(self.pretrained_model_path.glob('*.tar'))