import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timeit
import warnings

from tqdm import tqdm

from models.lstm.model import Model
from models.lstm.ntasgd import NTASGD

from scripts.data_handling import build_vocab
from scripts.plot import plot_loss

from torch.utils.data import DataLoader


class AwdLSTM:
    SEED = 12345

    def __init__(self, corpora, name, save_path, layer_num=3, embed_size=400, hidden_size=1150, lstm_type="pytorch",
                 w_drop=0.5, dropout_i=0.4, dropout_l=0.3, dropout_o=0.4, dropout_e=0.1, winit=0.1,
                 batch_size=40, valid_batch_size=10, bptt=70, ar=2, tar=1, weight_decay=1.2e-6,
                 epochs=750, lr=30, max_grad_norm=0.25, non_mono=5, device="gpu", log=50000, min_word_count=5,
                 max_vocab_size=None):
        self.corpora = corpora
        self.name = name
        self.save_path = save_path
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
        torch.autograd.set_detect_anomaly(True)

        self.set_device()

        self.save_path.mkdir(exist_ok=True, parents=True)
        self.log_file = self.set_log_file()

        self.data_init()

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
        log = open(str(self.save_path / f'{self.name}.csv'), "w")
        log.write("epoch,valid_ppl\n")
        return log

    def save_model(self, model):
        torch.save({'model_state_dict': model.state_dict()}, str(self.save_path / f'{self.name}.tar'))

    def chunk_examples(self, examples):
        text, fix_dur = [], []
        for sentence in examples['text']:
            text += sentence
        for sentence in examples['fix_dur']:
            fix_dur += sentence
        return {'text': text, 'fix_dur': fix_dur}

    def data_init(self):
        text = self.corpora
        split = text.corpora.train_test_split(test_size=0.2, seed=self.SEED)

        trn = split['train']
        vld = split['test']

        vocab = build_vocab(trn, self.min_word_count, self.max_vocab_size)[0]

        print("Numericalizing Training set")
        trn = trn.map(lambda row: {"text": vocab.forward(row["text"]), "fix_dur": row["fix_dur"]}, num_proc=12)
        
        print("Numericalizing Validation set")
        vld = vld.map(lambda row: {"text": vocab.forward(row["text"]), "fix_dur": row["fix_dur"]}, num_proc=12)

        print("Reshaping Training set")
        trn = trn.map(self.chunk_examples, batched=True, remove_columns=trn.column_names)

        print("Reshaping Validation set")
        vld = vld.map(self.chunk_examples, batched=True, remove_columns=vld.column_names)
        
        print("Loading training tokens...")
        self.trn_tokens = trn["text"].reshape(-1, 1)
        print("Loading training fix durations...")
        self.trn_fix = trn["fix_dur"].reshape(-1, 1)
        print("Loading validation tokens...")
        self.vld_tokens = vld["text"].reshape(-1, 1)
        print("Loading validation fix durations...")
        self.vld_fix = vld["fix_dur"].reshape(-1, 1)
        self.vocab = vocab.get_stoi()

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

    def perplexity_for_training(self, batches, model):
        model.eval()
        with torch.no_grad():
            losses = []
            batch_size = batches[0][0].size(1)
            states = model.state_init(batch_size)
            for x, y, _ in tqdm(batches, desc="Evaluating perplexity", leave=False):
                x = x.to(self.device)
                y = y.to(self.device)
                scores, states = model(x, states)
                loss = F.cross_entropy(scores, y.reshape(-1))
                losses.append(loss.data.item())
        return np.exp(np.mean(losses))

    def train_model(self, model, optimizer):
        tic = timeit.default_timer()
        total_words = 0
        print("Starting training.")
        best_val = 1e10
        loss_sg, loss_fix, ppl = [], [], {}
        for epoch in range(self.epochs):
            print("Epoch : {:d}".format(epoch + 1))
            seq_len = self.get_seq_len()
            optimizer.lr(seq_len / self.bptt * self.lr)
            states = model.state_init(self.batch_size)
            model.train()

            minibatches = self.minibatch(self.trn_tokens, self.trn_fix, seq_len)
            minibatches_for_trn = random.sample(minibatches, int(len(minibatches) * 0.1))
            for i, (x, y, fix) in tqdm(enumerate(minibatches), desc="Training", total=len(minibatches)):
                x = x.to(self.device)
                y = y.to(self.device)
                fix = fix.to(self.device)
                total_words += x.numel()
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
                
                #del x, y, fix, should_train_with_batch
                #gc.collect()

                if i % self.log == 0:
                    ppl[i] = self.perplexity_for_training(minibatches_for_trn, model)
                    model.train()

            tmp = {}
            for (prm, st) in optimizer.state.items():
                tmp[prm] = prm.clone().detach()
                prm.data = st['ax'].clone().detach()

            val_perp = self.perplexity(self.vld_tokens, self.vld_fix, model)
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
        plot_loss(loss_sg, loss_fix, self.name, self.save_path)
        plot_ppl(ppl, self.name, self.save_path)
        self.generate_embeddings(model)

    def generate_embeddings(self, model):
        weights = model.embed.W
        vocabulary = OrderedDict(sorted(self.vocab.items(), key=lambda x: x[1]))

        with open(str(self.save_path / f'{self.name}.vec'), "w") as f:
            f.write(f"{weights.shape[0]} {weights.shape[1]}\n")
            for _, (word, vector) in enumerate(zip(vocabulary.keys(), weights)):
                vector_str = ' '.join(str(x) for x in vector.tolist())
                f.write(f'{word} {vector_str}\n')

    def train(self):
        warnings.filterwarnings("ignore")
        self.trn_tokens, self.trn_fix = self.batchify(self.trn_tokens, self.trn_fix, self.batch_size)
        self.vld_tokens, self.vld_fix = self.batchify(self.vld_tokens, self.vld_fix, self.valid_batch_size)
        model = Model(len(self.vocab), self.embed_size, self.hidden_size, self.layer_num, self.w_drop, self.dropout_i,
                      self.dropout_l, self.dropout_o, self.dropout_e, self.winit, self.lstm_type)
        model.to(self.device)
        optimizer = NTASGD(model.parameters(), lr=self.lr, n=self.non_mono, weight_decay=self.weight_decay,
                           fine_tuning=False)
        self.train_model(model, optimizer)
