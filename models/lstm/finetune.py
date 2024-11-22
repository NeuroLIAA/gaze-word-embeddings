from models.lstm.main import AwdLSTM
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LinearLR
from scipy.stats import spearmanr
import timeit
import warnings

from models.lstm.model import Model
from models.lstm.ntasgd import NTASGD
from scripts.data_handling import chunk_examples


class AwdLSTMForFinetuning(AwdLSTM):
    def __init__(self, corpora, name, save_path, pretrained_model_path, stimuli_path, gaze_table, layer_num, embed_size,
                 hidden_size, lstm_type, w_drop, dropout_i, dropout_l, dropout_o, dropout_e, winit, batch_size,
                 valid_batch_size, bptt, ar, tar, weight_decay, epochs, lr, max_grad_norm, non_mono,
                 device, log, min_word_count, max_vocab_size, shard_count, pretrained_embeddings_path):
        super().__init__(corpora, name, save_path, pretrained_model_path, stimuli_path, gaze_table, layer_num,
                         embed_size, hidden_size, lstm_type, w_drop, dropout_i, dropout_l, dropout_o, dropout_e, winit,
                         batch_size, valid_batch_size, bptt, ar, tar, weight_decay, epochs, lr, max_grad_norm, non_mono,
                         device, log, min_word_count, max_vocab_size, shard_count, pretrained_embeddings_path)
        self.data_init()

    def set_log_dataset(self):
        self.log_dataset = pd.DataFrame({
            "epoch": pd.Series(dtype='int'),
            "lr": pd.Series(dtype='float'),
            "fix_corr": pd.Series(dtype='float'),
            "fix_pvalue": pd.Series(dtype='float'),
            "fix_std": pd.Series(dtype='float')
        })

    def log_data(self, epoch, lr, fix_corr, fix_pvalue, fix_std):
        self.log_dataset = self.log_dataset._append({
                "epoch": epoch,
                "lr": lr,
                "fix_corr": fix_corr,
                "fix_pvalue": fix_pvalue,
                "fix_std": fix_std
            }, ignore_index=True)

    def save_log(self):
        self.log_dataset.to_csv(self.save_path / f'{self.name}.csv', index=False)

    def data_init(self):
        data = self.corpora.corpora
        vocab = self.generate_vocab(data, self.pretrained_model_path / 'vocab.pt')
        print("Numericalizing Training set")
        data = data.map(
            lambda row: {"text": vocab(row["text"]), "fix_dur": row["fix_dur"]}, num_proc=12
        )
        print("Reshaping Training set")
        data = data.map(chunk_examples, batched=True, remove_columns=data.column_names, num_proc=12)
        data = data.with_format("torch")
        self.vocab = vocab
        self.data = data

    def train_model(self, model, optimizer, scheduler):
        tic = timeit.default_timer()
        print("Starting finetuning.")
        n_gaze_features = len(self.gaze_table.columns)
        metrics = {"loss_sg": [], "loss_fix": [],
                   "fix_corrs": [[] for _ in range(n_gaze_features)],
                   "fix_pvalues": [[] for _ in range(n_gaze_features)]}
        for epoch in range(self.epochs):
            print("Epoch : {:d}".format(epoch + 1))
            print("Learning rate : {:.3f}".format(scheduler.get_last_lr()[0]))
            self.train_epoch(model, optimizer, metrics)
            scheduler.step()

            self.save_model(model)
            toc = timeit.default_timer()
            print("Since beginning : {:.3f} mins".format(round((toc - tic) / 60)))
            print("*************************************************\n")
        self.plot_loss(metrics['loss_sg'], metrics['loss_fix']) 
        self.save_log()
        self.generate_embeddings(model)
        
    def train(self):
        warnings.filterwarnings("ignore")
        n_gaze_features = len(self.gaze_table.columns)
        model = Model(len(self.vocab), self.embed_size, self.hidden_size, n_gaze_features, self.layer_num, self.w_drop,
                      self.dropout_i, self.dropout_l, self.dropout_o, self.dropout_e, self.winit, self.lstm_type)
        model.to(self.device)
        checkpoint = torch.load(self.checkpoint())
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = NTASGD(model.parameters(), lr=self.lr, n=self.non_mono, weight_decay=self.weight_decay, fine_tuning=True)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=(0.3 / self.lr), total_iters=self.epochs)
        self.train_model(model, optimizer, scheduler)

    def checkpoint(self):
        # TODO: update regressor output dimension to n_gaze_features
        return next(self.pretrained_model_path.glob('*.tar'))
