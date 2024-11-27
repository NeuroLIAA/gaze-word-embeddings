from models.lstm.main import AwdLSTM
import pandas as pd
import torch
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
import timeit
import warnings
from models.lstm.model import Model
from models.lstm.ntasgd import NTASGD
from scripts.data_handling import get_vocab, perplexity, collate_fn_lstm
from scripts.utils import get_words_in_corpus


class AwdLSTMForTraining(AwdLSTM):

    def __init__(self, corpora, name, save_path, pretrained_model_path, stimuli_path, gaze_table, layer_num, embed_size,
                 hidden_size, lstm_type, w_drop, dropout_i, dropout_l, dropout_o, dropout_e, winit, batch_size,
                 valid_batch_size, bptt, ar, tar, weight_decay, epochs, lr, max_grad_norm, non_mono, 
                 device, log, min_word_count, max_vocab_size, pretrained_embeddings_path):
        super().__init__(corpora, name, save_path, pretrained_model_path, stimuli_path, gaze_table, layer_num,
                         embed_size, hidden_size, lstm_type, w_drop, dropout_i, dropout_l, dropout_o, dropout_e, winit,
                         batch_size, valid_batch_size, bptt, ar, tar, weight_decay, epochs, lr, max_grad_norm, non_mono,
                         device, log, min_word_count, max_vocab_size, pretrained_embeddings_path)
        splits = self.corpora.corpora.train_test_split(test_size=0.2, seed=self.SEED)
        self.data, self.val_data = splits['train'], splits['test']
        self.vocab = self.generate_vocab(self.data, self.save_path / 'vocab.pt')

    def set_log_dataset(self):
        self.log_dataset = pd.DataFrame({
            "epoch": pd.Series(dtype='int'),
            "valid_ppl": pd.Series(dtype='float'),
            "lr": pd.Series(dtype='float')
        })

    def log_data(self, epoch, valid_ppl, lr):
        self.log_dataset = self.log_dataset._append({
            "epoch": epoch,
            "valid_ppl": valid_ppl,
            "lr": lr
        }, ignore_index=True)

    def save_log(self):
        self.log_dataset.to_csv(self.save_path / f'{self.name}.csv', index=False)

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

    def train_model(self, model, dataloader, optimizer, scheduler):
        tic = timeit.default_timer()
        print("Starting training.")
        best_val = 1e10
        n_gaze_features = len(self.gaze_table.columns)
        val_dataloader = DataLoader(self.val_data,
                                    batch_size=self.batch_size * 5,
                                    shuffle=False,
                                    collate_fn=lambda batch: collate_fn_lstm(batch, self.batch_size, self.vocab,
                                                                             self.gaze_table),
                                    num_workers=8)
        metrics = {"loss_sg": [], "loss_fix": [], "perplexity": [],
                   "fix_corrs": [[] for _ in range(n_gaze_features)],
                   "fix_pvalues": [[] for _ in range(n_gaze_features)]}
        for epoch in range(self.epochs):
            print("Epoch : {:d}".format(epoch + 1))
            print("Learning rate : {:.3f}".format(scheduler.get_last_lr()[0]))
            self.train_epoch(model, dataloader, optimizer, metrics)
            tmp = {}
            for (prm, st) in optimizer.state.items():
                tmp[prm] = prm.clone().detach()
                prm.data = st['ax'].clone().detach()
            val_perp = perplexity(val_dataloader, model, self.batch_size, self.device)
            optimizer.check(val_perp)
            metrics['perplexity'].append(val_perp)
            if val_perp < best_val:
                best_val = val_perp
                print('Best validation perplexity : {:.3f}'.format(best_val))
                self.save_model(model)
                print('Model saved!')
            for (prm, st) in optimizer.state.items():
                prm.data = tmp[prm].clone().detach()
            self.log_data(epoch + 1, val_perp, scheduler.get_last_lr()[0])
            scheduler.step()
            toc = timeit.default_timer()
            print('Validation set perplexity : {:.3f}'.format(val_perp))
            print('Since beginning : {:.3f} mins'.format(round((toc - tic) / 60)))
            print('*************************************************\n')

        self.plot_loss(metrics['loss_sg'], metrics['loss_fix'])
        self.plot_perplexity(metrics['perplexity'])
        self.save_log()
        self.generate_embeddings(model)
        
    def train(self):
        warnings.filterwarnings("ignore")
        n_gaze_features = len(self.gaze_table.columns)
        model = Model(len(self.vocab), self.embed_size, self.hidden_size, n_gaze_features, self.layer_num, self.w_drop,
                      self.dropout_i, self.dropout_l, self.dropout_o, self.dropout_e, self.winit, self.lstm_type)
        model.to(self.device)
        self.load_pretrained_embeddings(model)
        dataloader = DataLoader(self.data,
                                batch_size=self.batch_size * 4,
                                shuffle=True,
                                collate_fn=lambda batch: collate_fn_lstm(batch, self.batch_size, self.vocab,
                                                                         self.gaze_table),
                                num_workers=8)
        optimizer = NTASGD(model.parameters(), lr=self.lr, n=self.non_mono, weight_decay=self.weight_decay,
                           fine_tuning=False)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=(0.3 / self.lr), total_iters=self.epochs)
        self.train_model(model, dataloader, optimizer, scheduler)
