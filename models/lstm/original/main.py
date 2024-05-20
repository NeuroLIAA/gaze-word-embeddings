import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import timeit
import warnings
import pickle
from fastai.text.all import *
import tqdm
from chars import CharRemover
from model import Model
from ntasgd import NTASGD
import sys

# Replace '/path/to/directory' with the actual path
sys.path.append('./scripts')

from corpora import Corpora
from data_handling import build_vocab

class AWD_LSTM_Model:
    def __init__(self, name, layer_num = 3, embed_size = 400, hidden_size = 1150, lstm_type = "pytorch", 
                 w_drop = 0.5, dropout_i = 0.4, dropout_l = 0.3, dropout_o = 0.4, dropout_e = 0.1, winit = 0.1,
                 batch_size = 40, valid_batch_size = 10, bptt = 70, ar = 2, tar = 1, weight_decay = 1.2e-6, 
                 epochs = 750, lr = 30, max_grad_norm = 0.25, non_mono = 5, device = "gpu", log = 100):
        self.name = name
        self.filename = "./models/lstm/data/models/" + self.name + ".tar"
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
        
        self.set_device()
        
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
            
    def set_log_file(self):
        log = open("./models/lstm/data/" + self.name + ".csv", "w")
        log.write("epoch,valid_ppl\n")
        return log
        
    def save_model(self, model):
        torch.save({'model_state_dict': model.state_dict()}, self.filename)
        
    def data_init(self, min_token_len = 1, max_token_len = 20, min_sentence_len = 5, min_word_count = 5, max_vocab_size = None):
        text = Corpora(min_token_len, max_token_len, min_sentence_len, True)
        text.add_corpus('all_wikis', 'remote', 0.01, 1)
        split = text.corpora.train_test_split(test_size=0.2, seed=12345)
        
        trn = split['train']
        vld = split['test']
            
        vocab = build_vocab(trn, min_word_count, max_vocab_size)[0]

        vocab_size = len(vocab.get_stoi())
        with open("./models/lstm/data/" + self.name + ".json", 'w') as f:
            json.dump(vocab.get_stoi(), f)
            
        print("Numericalizing Training set")
        trn = trn.map(lambda row: {"text": vocab.forward(row["text"])}, num_proc=12)
        trn = [token for text in tqdm.tqdm(trn['text']) for token in text]
        
        print("Numericalizing Validation set")
        vld = vld.map(lambda row: {"text": vocab.forward(row["text"])}, num_proc=12)
        vld = [token for text in tqdm.tqdm(vld['text']) for token in text]
        
        self.trn = torch.tensor(trn,dtype=torch.int64).reshape(-1, 1)
        self.vld = torch.tensor(vld,dtype=torch.int64).reshape(-1, 1)
        self.vocab_size = vocab_size

    def get_seq_len(self):
        seq_len = self.bptt if np.random.random() < 0.95 else self.bptt/2
        seq_len = round(np.random.normal(seq_len, 5))
        while seq_len <= 5 or seq_len >= 90:
            seq_len = self.bptt if np.random.random() < 0.95 else self.bptt/2
            seq_len = round(np.random.normal(seq_len, 5))
        return seq_len

    def batchify(self, data, batch_size):
        num_batches = data.size(0) // batch_size
        data = data[:num_batches * batch_size]
        return data.reshape(batch_size, -1).transpose(1, 0)

    def minibatch(self, data, seq_length):
        num_batches = data.size(0)
        dataset = []
        for i in range(0, num_batches-1, seq_length):
            ls = min(i+seq_length, num_batches-1)
            x = data[i:ls,:]
            y = data[i+1:ls+1,:]
            dataset.append((x, y))
        return dataset

    def perplexity(self, data, model):
        model.eval()
        data = self.minibatch(data, self.bptt)
        with torch.no_grad():
            losses = []
            batch_size = data[0][0].size(1)
            states = model.state_init(batch_size)
            for x, y in data:
                x = x.to(self.device)
                y = y.to(self.device)
                scores, states = model(x, states)
                loss = F.cross_entropy(scores, y.reshape(-1))
                losses.append(loss.data.item())
        return np.exp(np.mean(losses))

    def train(self, model, optimizer):
        tic = timeit.default_timer()
        total_words = 0
        print("Starting training.")
        best_val = 1e10
        try:
            for epoch in range(self.epochs):        
                seq_len = self.get_seq_len()
                num_batch = ((self.trn.size(0) - 1) // seq_len + 1)
                optimizer.lr(seq_len / self.bptt * self.lr)
                states = model.state_init(self.batch_size)
                model.train()
                for i, (x, y) in enumerate(self.minibatch(self.trn, seq_len)):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    total_words += x.numel()
                    states = model.detach(states)
                    scores, states, activations = model(x, states)
                    loss = F.cross_entropy(scores, y.reshape(-1))
                    h, h_m = activations
                    ar_reg = self.ar * h_m.pow(2).mean()
                    tar_reg = self.tar * (h[:-1] - h[1:]).pow(2).mean()
                    loss_reg = loss + ar_reg + tar_reg
                    loss_reg.backward()
                    norm = nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    if i % (self.log) == 0:
                        toc = timeit.default_timer()
                        print("batch no = {:d} / {:d}, ".format(i, num_batch) +
                            "train loss = {:.3f}, ".format(loss.item()) +
                            "ar val = {:.3f}, ".format(ar_reg.item()) + 
                            "tar val = {:.3f}, ".format(tar_reg.item()) + 
                            "wps = {:d}, ".format(round(total_words / (toc - tic))) +
                            "dw.norm() = {:.3f}, ".format(norm) +
                            "lr = {:.3f}, ".format(seq_len / self.bptt * self.lr) + 
                            "since beginning = {:d} mins, ".format(round((toc - tic) / 60)) +
                            "cuda memory = {:.3f} GBs".format(torch.cuda.max_memory_allocated()/1024/1024/1024))
        
                tmp = {}
                for (prm,st) in optimizer.state.items():
                    tmp[prm] = prm.clone().detach()
                    prm.data = st['ax'].clone().detach()
        
                val_perp = self.perplexity(self.vld, model)
                optimizer.check(val_perp)

                self.log_file.write("{:d},{:.3f}\n".format(epoch+1, val_perp))
        
                if val_perp < best_val:
                    best_val = val_perp
                    print("Best validation perplexity : {:.3f}".format(best_val))
                    self.save_model(model)
                    print("Model saved!")
        
                for (prm, st) in optimizer.state.items():
                    prm.data = tmp[prm].clone().detach()
                    
                print("Epoch : {:d} || Validation set perplexity : {:.3f}".format(epoch+1, val_perp))
                print("*************************************************\n")        
        except KeyboardInterrupt:
            print("Finishing training early.")
        self.log_file.close()
        checkpoint = torch.load(self.filename)
        model.load_state_dict(checkpoint['model_state_dict'])

    def start_training(self):
        warnings.filterwarnings("ignore")
        self.trn = self.batchify(self.trn, self.batch_size)
        self.vld = self.batchify(self.vld, self.valid_batch_size)
        model = Model(self.vocab_size, self.embed_size, self.hidden_size, self.layer_num, self.w_drop, self.dropout_i, 
                    self.dropout_l, self.dropout_o, self.dropout_e, self.winit, self.lstm_type)
        model.to(self.device)
        optimizer = NTASGD(model.parameters(), lr=self.lr, n=self.non_mono, weight_decay=self.weight_decay, fine_tuning=False)
        self.train(model, optimizer)

model = AWD_LSTM_Model("pytorch_spanish_spacy_30000", epochs=200)
model.data_init(max_vocab_size=30000)
model.start_training()