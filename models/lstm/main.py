from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from scipy.stats import spearmanr
from tqdm import tqdm
from pathlib import Path
from scripts.data_handling import get_vocab, batchify, minibatch
from scripts.utils import get_words_in_corpus
from scripts.plot import plot_loss


class AwdLSTM:
    @staticmethod
    def create_from_args(corpora, name, save_path, pretrained_model_path, stimuli_path, gaze_table, embed_size=300,
                         batch_size=40, epochs=50, lr=30, min_word_count=20, max_vocab_size=None,
                         pretrained_embeddings_path=None):
        pretrained_embeddings_path = Path(pretrained_embeddings_path) if pretrained_embeddings_path else None
        params = {'corpora': corpora, 'name': name, 'save_path': save_path,
                  'pretrained_model_path': pretrained_model_path, 'stimuli_path': stimuli_path,
                  'gaze_table': gaze_table, 'layer_num': 3, 'embed_size': embed_size, 'hidden_size': 1150,
                  'lstm_type': "pytorch", 'w_drop': 0.5, 'dropout_i': 0.4, 'dropout_l': 0.3, 'dropout_o': 0.4,
                  'dropout_e': 0.1, 'winit': 0.1, 'batch_size': batch_size, 'valid_batch_size': 10, 'bptt': 70,
                  'ar': 2, 'tar': 1, 'weight_decay': 1.2e-6, 'epochs': epochs, 'lr': lr, 'max_grad_norm': 0.25,
                  'non_mono': 5, 'device': "gpu", 'log': 50000, 'min_word_count': min_word_count,
                  'max_vocab_size': max_vocab_size, 'shard_count': 5,
                  'pretrained_embeddings_path': pretrained_embeddings_path}

        if pretrained_model_path:
            from models.lstm.finetune import AwdLSTMForFinetuning
            params['shard_count'] = 1
            return AwdLSTMForFinetuning(**params)
        else:
            from models.lstm.train import AwdLSTMForTraining
            return AwdLSTMForTraining(**params)
        
    SEED = 12345

    def __init__(self, corpora, name, save_path, pretrained_model_path, stimuli_path, gaze_table, layer_num=3,
                 embed_size=300, hidden_size=1150, lstm_type="pytorch", w_drop=0.5, dropout_i=0.4, dropout_l=0.3,
                 dropout_o=0.4, dropout_e=0.1, winit=0.1, batch_size=40, valid_batch_size=10, bptt=70, ar=2, tar=1,
                 weight_decay=1.2e-6, epochs=750, lr=30, max_grad_norm=0.25, non_mono=5, device="gpu", log=50000,
                 min_word_count=5, max_vocab_size=None, shard_count=5, pretrained_embeddings_path=None):
        self.corpora = corpora
        self.name = name
        self.save_path = save_path
        self.pretrained_model_path = pretrained_model_path
        self.stimuli_path = stimuli_path
        self.gaze_table = gaze_table
        self.layer_num = layer_num
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.lstm_type = lstm_type
        self.w_drop = w_drop
        self.dropout_i = dropout_i
        self.dropout_l = dropout_l
        self.dropout_o = dropout_o
        self.dropout_e = dropout_e
        self.winit = winit
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.bptt = bptt
        self.ar = ar
        self.tar = tar
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.non_mono = non_mono
        self.device = device
        self.log = log
        self.min_word_count = min_word_count
        self.max_vocab_size = max_vocab_size
        self.shard_count = shard_count
        self.pretrained_embeddings_path = Path(pretrained_embeddings_path) if pretrained_embeddings_path else None
        self.set_device()
        self.save_path.mkdir(exist_ok=True, parents=True)
        self.set_log_dataset()

    def set_device(self):
        if self.device == "gpu" and torch.cuda.is_available():
            print("Model will be training on the GPU.\n")
            self.device = torch.device('cuda')
        elif self.device == "gpu":
            print("No GPU detected. Falling back to CPU.\n")
            self.device = torch.device('cpu')
        else:
            print("Model will be training on the CPU.\n")
            self.device = torch.device('cpu')

    def plot_loss(self, loss_sg, loss_fix):
        plot_loss(loss_sg, loss_fix, self.name, self.save_path, 'LSTM')

    def generate_embeddings(self, model):
        weights = model.embed.W
        vocabulary = OrderedDict(sorted(self.vocab.get_stoi().items(), key=lambda x: x[1]))

        with open(str(self.save_path / f'{self.name}.vec'), "w") as f:
            f.write(f"{weights.shape[0]} {weights.shape[1]}\n")
            for _, (word, vector) in enumerate(zip(vocabulary.keys(), weights)):
                vector_str = ' '.join(str(x) for x in vector.tolist())
                f.write(f'{word} {vector_str}\n')

    def save_model(self, model):
        torch.save({'model_state_dict': model.state_dict()}, str(self.save_path / f'{self.name}.tar'))

    def generate_vocab(self, data, vocab_savepath=None):
        words_in_stimuli = get_words_in_corpus(self.stimuli_path)
        return get_vocab(corpora=data,
                         min_count=self.min_word_count,
                         words_in_stimuli=words_in_stimuli,
                         max_vocab_size=self.max_vocab_size,
                         is_baseline=self.pretrained_model_path is None,
                         vocab_savepath=vocab_savepath)[0]

    def get_seq_len(self):
        seq_len = self.bptt if np.random.random() < 0.95 else self.bptt / 2
        seq_len = round(np.random.normal(seq_len, 5))
        while seq_len <= 5 or seq_len >= 90:
            seq_len = self.bptt if np.random.random() < 0.95 else self.bptt / 2
            seq_len = round(np.random.normal(seq_len, 5))
        return seq_len

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
    
    def train_epoch(self, model, optimizer, metrics):
        seq_len = self.get_seq_len()
        states = model.state_init(self.batch_size)
        n_gaze_features = len(self.gaze_table.columns)
        model.train()
        for i in tqdm(range(self.shard_count), desc="Sharding"):
            dataset = self.data.shard(num_shards=self.shard_count, index=i, contiguous=True)
            tokens = dataset["text"].reshape(-1, 1)
            words = [self.vocab.get_word(x) for x in dataset['text']]
            fix_labels = torch.FloatTensor(self.gaze_table.reindex(words, fill_value=0).values)
            tokens, fix_dur = batchify(tokens, fix_labels, self.batch_size)
            minibatches = minibatch(tokens, fix_dur, seq_len)
            for _, (x, y, fix) in tqdm(enumerate(minibatches), desc="Training Shard N°{:d}".format(i+1),
                                       total=len(minibatches)):
                x = x.to(self.device)
                y = y.to(self.device)
                fix = fix.to(self.device).reshape(-1, n_gaze_features)
                states = model.detach(states)
                scores, states, activations, fix_preds = model(x, states)

                loss = cross_entropy(scores, y.reshape(-1))
                h, h_m = activations
                ar_reg = self.ar * h_m.pow(2).mean()
                tar_reg = self.tar * (h[:-1] - h[1:]).pow(2).mean()

                if fix.sum() > 0:
                    if n_gaze_features == 1:
                        fix_preds = fix_preds.unsqueeze(dim=1)
                    fix_loss = torch.nn.L1Loss()(fix_preds, fix)
                    fix_preds = fix_preds.cpu().detach().numpy()
                    fix_labels = fix.cpu().detach().numpy()
                    batch_correlations = [spearmanr(fix_preds[:, k], fix_labels[:, k]) for k in range(n_gaze_features)]
                    for j in range(n_gaze_features):
                        metrics['fix_corrs'][j].append(batch_correlations[j].correlation)
                        metrics['fix_pvalues'][j].append(batch_correlations[j].pvalue)
                else:
                    fix_loss = torch.tensor(0.0)

                loss_reg = loss + ar_reg + tar_reg + fix_loss
                loss_reg.backward()
                metrics['loss_sg'].append(loss.item() + ar_reg.item() + tar_reg.item())
                metrics['loss_fix'].append(fix_loss.item())
                nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
