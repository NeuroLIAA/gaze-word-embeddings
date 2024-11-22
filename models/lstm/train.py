from models.lstm.main import AwdLSTM
import pandas as pd
import torch
from torch.optim.lr_scheduler import LinearLR
import timeit
import warnings
from models.lstm.model import Model
from models.lstm.ntasgd import NTASGD
from scripts.data_handling import get_vocab, batchify, chunk_examples, perplexity
from scripts.utils import get_words_in_corpus


class AwdLSTMForTraining(AwdLSTM):

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
        data = data.map(chunk_examples, batched=True, remove_columns=data.column_names)
        print("Reshaping Validation set")
        vld_data = vld_data.map(chunk_examples, batched=True, remove_columns=vld_data.column_names)
        self.vocab = vocab
        
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

    def train_model(self, model, optimizer, scheduler):
        tic = timeit.default_timer()
        print("Starting training.")
        best_val = 1e10
        n_gaze_features = len(self.gaze_table.columns)
        metrics = {"loss_sg": [], "loss_fix": [],
                   "fix_corrs": [[] for _ in range(n_gaze_features)],
                   "fix_pvalues": [[] for _ in range(n_gaze_features)]}
        for epoch in range(self.epochs):
            print("Epoch : {:d}".format(epoch + 1))
            print("Learning rate : {:.3f}".format(scheduler.get_last_lr()[0]))
            self.train_epoch(model, optimizer, metrics)
            tmp = {}
            for (prm, st) in optimizer.state.items():
                tmp[prm] = prm.clone().detach()
                prm.data = st['ax'].clone().detach()
            val_perp = perplexity(self.bptt, self.vld_data_tokens, self.vld_data_fix, model, self.device)
            optimizer.check(val_perp)
            if val_perp < best_val:
                best_val = val_perp
                print("Best validation perplexity : {:.3f}".format(best_val))
                self.save_model(model)
                print("Model saved!")
            for (prm, st) in optimizer.state.items():
                prm.data = tmp[prm].clone().detach()
            self.log_data(epoch + 1, val_perp, scheduler.get_last_lr()[0])
            scheduler.step()
            toc = timeit.default_timer()
            print("Validation set perplexity : {:.3f}".format(val_perp))
            print("Since beginning : {:.3f} mins".format(round((toc - tic) / 60)))
            print("*************************************************\n")

        self.plot_loss(metrics['loss_sg'], metrics['loss_fix'])
        self.save_log()
        self.generate_embeddings(model)
        
    def train(self):
        warnings.filterwarnings("ignore")
        self.vld_data_tokens, self.vld_data_fix = batchify(self.vld_data_tokens, self.vld_data_fix,
                                                           self.valid_batch_size)
        n_gaze_features = len(self.gaze_table.columns)
        model = Model(len(self.vocab), self.embed_size, self.hidden_size, n_gaze_features, self.layer_num, self.w_drop,
                      self.dropout_i, self.dropout_l, self.dropout_o, self.dropout_e, self.winit, self.lstm_type)
        model.to(self.device)
        self.load_pretrained_embeddings(model)
        optimizer = NTASGD(model.parameters(), lr=self.lr, n=self.non_mono, weight_decay=self.weight_decay,
                           fine_tuning=False)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=(0.3 / self.lr), total_iters=self.epochs)
        self.train_model(model, optimizer, scheduler)
