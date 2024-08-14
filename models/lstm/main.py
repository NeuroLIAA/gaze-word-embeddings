from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timeit
import warnings
import random

from tqdm import tqdm

from models.lstm.model import Model
from models.lstm.ntasgd import NTASGD

from scripts.data_handling import get_vocab
from scripts.utils import get_words_in_corpus
from scripts.plot import plot_loss

class AwdLSTM:
    @staticmethod
    def create_from_args(corpora, name, save_path, pretrained_path, stimuli_path, embed_size=400, batch_size=40, 
                         epochs=750, lr=30, min_word_count=5, max_vocab_size=None):
        if pretrained_path:
            return AwdLSTMForFinetuning(corpora, name, save_path, pretrained_path, stimuli_path, layer_num=3,
                                        embed_size=embed_size, hidden_size=1150, lstm_type="pytorch", w_drop=0.5,
                                        dropout_i=0.4, dropout_l=0.3, dropout_o=0.4, dropout_e=0.1, winit=0.1,
                                        batch_size=batch_size, valid_batch_size=10, bptt=70, ar=2, tar=1,
                                        weight_decay=1.2e-6, epochs=epochs, lr=lr, max_grad_norm=0.25, non_mono=5,
                                        device="gpu", log=50000, min_word_count=min_word_count, max_vocab_size=max_vocab_size,
                                        shard_count=5)
        else:
            return AwdLSTMForTraining(corpora, name, save_path, pretrained_path, stimuli_path, layer_num=3,
                                      embed_size=embed_size, hidden_size=1150, lstm_type="pytorch", w_drop=0.5,
                                      dropout_i=0.4, dropout_l=0.3, dropout_o=0.4, dropout_e=0.1, winit=0.1,
                                      batch_size=batch_size, valid_batch_size=10, bptt=70, ar=2, tar=1,
                                      weight_decay=1.2e-6, epochs=epochs, lr=lr, max_grad_norm=0.25, non_mono=5,
                                      device="gpu", log=50000, min_word_count=min_word_count, max_vocab_size=max_vocab_size,
                                      shard_count=20)
        
    SEED = 12345

    def __init__(self, corpora, name, save_path, pretrained_path, stimuli_path, layer_num=3, embed_size=400, hidden_size=1150, lstm_type="pytorch",
                 w_drop=0.5, dropout_i=0.4, dropout_l=0.3, dropout_o=0.4, dropout_e=0.1, winit=0.1,
                 batch_size=40, valid_batch_size=10, bptt=70, ar=2, tar=1, weight_decay=1.2e-6,
                 epochs=750, lr=30, max_grad_norm=0.25, non_mono=5, device="gpu", log=50000, min_word_count=5,
                 max_vocab_size=None, shard_count=5):
        self.corpora = corpora
        self.name = name
        self.save_path = save_path
        self.pretrained_path = pretrained_path
        self.stimuli_path = stimuli_path
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
        torch.autograd.set_detect_anomaly(True)

        self.set_device()

        self.save_path.mkdir(exist_ok=True, parents=True)
        self.log_file = self.set_log_file()

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

    def chunk_examples(self, examples):
        text, fix_dur = [], []
        for sentence in examples['text']:
            text += sentence
        for sentence in examples['fix_dur']:
            fix_dur += sentence
        return {'text': text, 'fix_dur': fix_dur}

    def get_seq_len(self):
        seq_len = self.bptt if np.random.random() < 0.95 else self.bptt / 2
        seq_len = round(np.random.normal(seq_len, 5))
        while seq_len <= 5 or seq_len >= 90:
            seq_len = self.bptt if np.random.random() < 0.95 else self.bptt / 2
            seq_len = round(np.random.normal(seq_len, 5))
        return seq_len

    def batchify(self, data, fix_data, batch_size):
        num_batches = data.size(0) // batch_size
        data = data[:num_batches * batch_size]
        fix_data = fix_data[:num_batches * batch_size]
        return data.reshape(batch_size, -1).transpose(1, 0), fix_data.reshape(batch_size, -1).transpose(1, 0)

    def minibatch(self, data, fix_data, seq_length):
        num_batches = data.size(0)
        dataset = []
        for i in range(0, num_batches - 1, seq_length):
            ls = min(i + seq_length, num_batches - 1)
            x = data[i:ls, :]
            fix_x = fix_data[i:ls, :]
            y = data[i + 1:ls + 1, :]
            dataset.append((x, y, fix_x))
        return dataset

    def perplexity(self, data, fix_data, model):
        model.eval()
        data = self.minibatch(data, fix_data, self.bptt)
        with torch.no_grad():
            losses = []
            batch_size = data[0][0].size(1)
            states = model.state_init(batch_size)
            for x, y, _ in tqdm(data, desc="Evaluating perplexity"):
                x = x.to(self.device)
                y = y.to(self.device)
                scores, states = model(x, states)
                loss = F.cross_entropy(scores, y.reshape(-1))
                losses.append(loss.data.item())
        return np.exp(np.mean(losses))
    
    def train_model(self, model, optimizer):
        tic = timeit.default_timer()
        print("Starting training.")
        best_val = 1e10
        loss_sg, loss_fix = [], []
        for epoch in range(self.epochs):
            print("Epoch : {:d}".format(epoch + 1))
            seq_len = self.get_seq_len()
            optimizer.lr(seq_len / self.bptt * self.lr)
            states = model.state_init(self.batch_size)
            model.train()
            
            for i in tqdm(range(self.shard_count), desc="Sharding"):
                dataset = self.data.shard(num_shards=self.shard_count, index=i, contiguous=True)
                
                print("Loading training tokens...")
                tokens = dataset["text"].reshape(-1, 1)
                print("Loading training fix durations...")
                fix_dur = dataset["fix_dur"].reshape(-1, 1)
                
                tokens, fix_dur = self.batchify(tokens, fix_dur, self.batch_size)
                
                minibatches = self.minibatch(tokens, fix_dur, seq_len)

                for j, (x, y, fix) in tqdm(enumerate(minibatches), desc="Training Shard N°{:d}".format(i+1), total=len(minibatches)):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    fix = fix.to(self.device)
                    states = model.detach(states)
                    scores, states, activations, fix_pred = model(x, states)

                    loss = F.cross_entropy(scores, y.reshape(-1))
                    h, h_m = activations
                    ar_reg = self.ar * h_m.pow(2).mean()
                    tar_reg = self.tar * (h[:-1] - h[1:]).pow(2).mean()

                    if fix.sum() > 0:
                        fix_loss = torch.nn.L1Loss()(fix_pred, fix.reshape(-1))
                    else:
                        fix_loss = torch.tensor(0.0)

                    loss_reg = loss + ar_reg + tar_reg + fix_loss
                    loss_reg.backward()

                    loss_sg.append(loss.item() + ar_reg.item() + tar_reg.item())
                    loss_fix.append(fix_loss.item())

                    nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

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

            toc = timeit.default_timer()
            print("Validation set perplexity : {:.3f}".format(val_perp))
            print("Since beginning : {:.3f} mins".format(round((toc - tic) / 60)))
            print("lr : {:.3f}".format(seq_len / self.bptt * self.lr))
            print("*************************************************\n")
        self.log_file.close()
        self.plot_loss(loss_sg, loss_fix)
        self.generate_embeddings(model)
    
    def data_init(self):
        text = self.corpora
        split = text.corpora.train_test_split(test_size=0.2, seed=self.SEED)

        data = split['train']
        vld_data = split['test']

        vocab = self.generate_vocab(data)

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

class AwdLSTMForTraining(AwdLSTM):
    def __init__(self, corpora, name, save_path, pretrained_path, stimuli_path, layer_num, embed_size, hidden_size, 
                 lstm_type, w_drop, dropout_i, dropout_l, dropout_o, dropout_e, winit, batch_size, 
                 valid_batch_size, bptt, ar, tar, weight_decay, epochs, lr, max_grad_norm, non_mono, 
                 device, log, min_word_count, max_vocab_size, shard_count):
        super().__init__(corpora, name, save_path, pretrained_path, stimuli_path, layer_num, embed_size, hidden_size, 
                         lstm_type, w_drop, dropout_i, dropout_l, dropout_o, dropout_e, winit, batch_size, valid_batch_size, 
                         bptt, ar, tar, weight_decay, epochs, lr, max_grad_norm, non_mono, device, log, min_word_count, max_vocab_size, 
                         shard_count)
        self.data_init()
        
    def generate_vocab(self, data):
        words_in_stimuli = get_words_in_corpus(self.stimuli_path)
        vocab_savepath = self.save_path / 'vocab.pt'
        return get_vocab(corpora=data, 
                          min_count=self.min_word_count, 
                          words_in_stimuli=words_in_stimuli, 
                          max_vocab_size=self.max_vocab_size, 
                          is_baseline=self.pretrained_path is None,
                          vocab_savepath=vocab_savepath)[0]
        
    def set_log_file(self):
        log = open(str(self.save_path / f'{self.name}.csv'), "w")
        log.write("epoch,valid_ppl\n")
        return log
        
    def save_model(self, model):
        torch.save({'model_state_dict': model.state_dict()}, str(self.save_path / f'{self.name}.tar'))
        
    def generate_embeddings(self, model):
        weights = model.embed.W
        vocabulary = OrderedDict(sorted(self.vocab.items(), key=lambda x: x[1]))

        with open(str(self.save_path / f'{self.name}.vec'), "w") as f:
            f.write(f"{weights.shape[0]} {weights.shape[1]}\n")
            for _, (word, vector) in enumerate(zip(vocabulary.keys(), weights)):
                vector_str = ' '.join(str(x) for x in vector.tolist())
                f.write(f'{word} {vector_str}\n')
                
    def plot_loss(self, loss_sg, loss_fix):
        plot_loss(loss_sg, loss_fix, self.name, self.save_path)
        
    def train(self):
        warnings.filterwarnings("ignore")
        self.vld_data_tokens, self.vld_data_fix = self.batchify(self.vld_data_tokens, self.vld_data_fix, self.valid_batch_size)
        model = Model(len(self.vocab), self.embed_size, self.hidden_size, self.layer_num, self.w_drop, self.dropout_i,
                      self.dropout_l, self.dropout_o, self.dropout_e, self.winit, self.lstm_type)
        model.to(self.device)
        optimizer = NTASGD(model.parameters(), lr=self.lr, n=self.non_mono, weight_decay=self.weight_decay, fine_tuning=False)
        self.train_model(model, optimizer)
        
class AwdLSTMForFinetuning(AwdLSTM):
    def __init__(self, corpora, name, save_path, pretrained_path, stimuli_path, layer_num, embed_size, hidden_size, 
                 lstm_type, w_drop, dropout_i, dropout_l, dropout_o, dropout_e, winit, batch_size, 
                 valid_batch_size, bptt, ar, tar, weight_decay, epochs, lr, max_grad_norm, non_mono, 
                 device, log, min_word_count, max_vocab_size, shard_count):
        super().__init__(corpora, name, save_path, pretrained_path, stimuli_path, layer_num, embed_size, hidden_size, 
                         lstm_type, w_drop, dropout_i, dropout_l, dropout_o, dropout_e, winit, batch_size, valid_batch_size, 
                         bptt, ar, tar, weight_decay, epochs, lr, max_grad_norm, non_mono, device, log, min_word_count, max_vocab_size, 
                         shard_count)
        self.data_init()
        
    def generate_vocab(self, data):
        words_in_stimuli = get_words_in_corpus(self.stimuli_path)
        vocab_savepath = self.pretrained_path.parent / 'vocab.pt'
        return get_vocab(corpora=data, 
                          min_count=self.min_word_count, 
                          words_in_stimuli=words_in_stimuli, 
                          max_vocab_size=self.max_vocab_size, 
                          is_baseline=self.pretrained_path is None,
                          vocab_savepath=vocab_savepath)[0]
        
    def set_log_file(self):
        name = self.name + "_finetuned"
        log = open(str(self.save_path / f'{name}.csv'), "w")
        log.write("epoch,valid_ppl\n")
        return log
    
    def generate_embeddings(self, model):
        weights = model.embed.W
        vocabulary = OrderedDict(sorted(self.vocab.items(), key=lambda x: x[1]))
        name = self.name + "_finetuned"

        with open(str(self.save_path / f'{name}.vec'), "w") as f:
            f.write(f"{weights.shape[0]} {weights.shape[1]}\n")
            for _, (word, vector) in enumerate(zip(vocabulary.keys(), weights)):
                vector_str = ' '.join(str(x) for x in vector.tolist())
                f.write(f'{word} {vector_str}\n')
        
    def save_model(self, model):
        name = self.name + "_finetuned"
        torch.save({'model_state_dict': model.state_dict()}, str(self.save_path / f'{name}.tar'))
        
    def train(self):
        warnings.filterwarnings("ignore")
        model = Model(len(self.vocab), self.embed_size, self.hidden_size, self.layer_num, self.w_drop, self.dropout_i,
                      self.dropout_l, self.dropout_o, self.dropout_e, self.winit, self.lstm_type)
        model.to(self.device)
        checkpoint = torch.load(self.pretrained_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = NTASGD(model.parameters(), lr=self.lr, n=self.non_mono, weight_decay=self.weight_decay, fine_tuning=True)
        self.train_model(model, optimizer)
        
    def plot_loss(self, loss_sg, loss_fix):
        name = self.name + "_finetuned"
        plot_loss(loss_sg, loss_fix, name, self.save_path)