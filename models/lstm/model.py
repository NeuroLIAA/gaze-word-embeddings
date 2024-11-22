import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings


class Embed(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0, winit=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.dropout = dropout
        self.W = nn.Parameter(torch.Tensor(vocab_size, embed_size))
        self.reset_parameters(winit)
    
    def reset_parameters(self, winit):
        nn.init.uniform_(self.W, -winit, winit)

    def mask(self):
        return torch.empty(self.vocab_size, 1, device = self.W.device).bernoulli_(1-self.dropout)/(1-self.dropout)
                     
    def forward(self, x):
        embeddings = self.W[x]
        if self.training:
            mask = self.mask()[x]
            embeddings.mul_(mask)
        return embeddings

    def __repr__(self):
        return "Embedding(vocab: {}, embedding: {}, dropout: {})".format(self.vocab_size, self.embed_size, self.dropout)


class VariationalDropout(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input_, dropout):
        if self.training:
            mask = torch.empty(input_.size(1), input_.size(2), device=input_.device).bernoulli_(1-dropout)/(1-dropout)
            mask = mask.expand_as(input_)
            return mask * input_
        else:
            return input_
    
    def __repr__(self):
        return "VariationalDropout()"


class WeightDropLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, weight_drop=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_drop = weight_drop
        self.module = nn.LSTM(input_size, hidden_size)
        self.weight_name = 'weight_hh_l0'
        w = getattr(self.module, self.weight_name)
        self.register_parameter(f'{self.weight_name}_raw', nn.Parameter(w.clone().detach()))
        raw_w = getattr(self, f'{self.weight_name}_raw')
        self.reset_parameters()
        self.module._parameters[self.weight_name] = F.dropout(raw_w, p=self.weight_drop, training=False)
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)

    def __repr__(self):
        return "WeightDropLSTM(input: {}, hidden: {}, weight drop: {})".format(self.input_size, self.hidden_size, self.weight_drop)

    def _setweights(self):
        raw_w = getattr(self, f'{self.weight_name}_raw')
        self.module._parameters[self.weight_name] = F.dropout(raw_w, p=self.weight_drop, training=self.training)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            # To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)


class WeightDropLSTMCustom(nn.Module):
    def __init__(self, input_size, hidden_size, weight_drop = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_drop = weight_drop
        self.W_x = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.W_h = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.b_x = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.W_x, -stdv, stdv)
        nn.init.uniform_(self.W_h, -stdv, stdv)
        nn.init.uniform_(self.b_x, -stdv, stdv)
        nn.init.uniform_(self.b_h, -stdv, stdv)

    def __repr__(self):
        return "WeightDropLSTM(input: {}, hidden: {}, weight drop: {})".format(self.input_size, self.hidden_size, self.weight_drop)

    @staticmethod
    def lstm_step(x, h, c, W_x, W_h, b_x, b_h):
        gx = torch.addmm(b_x, x, W_x.t())
        gh = torch.addmm(b_h, h, W_h.t())
        xi, xf, xo, xn = gx.chunk(4, 1)
        hi, hf, ho, hn = gh.chunk(4, 1)
        inputgate = torch.sigmoid(xi + hi)
        forgetgate = torch.sigmoid(xf + hf)
        outputgate = torch.sigmoid(xo + ho)
        newgate = torch.tanh(xn + hn)
        c = forgetgate * c + inputgate * newgate
        h = outputgate * torch.tanh(c)
        return h, c

    # Takes input tensor x with dimensions: [T, B, X].
    def forward(self, input_, states):
        h, c = states
        outputs = []
        inputs = input_.unbind(0)
        W_h = F.dropout(self.W_h, self.weight_drop, self.training)
        for x_t in inputs:
            h, c = self.lstm_step(x_t, h, c, self.W_x, W_h, self.b_x, self.b_h)
            outputs.append(h)
        return torch.stack(outputs), (h, c)


class FCTied(nn.Module):
    def __init__(self, input_size, hidden_size, weight):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = weight
        self.b = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.b, -stdv, stdv)

    def forward(self, x):
        z = torch.addmm(self.b, x.view(-1, x.size(2)), self.W.t())
        return z

    def __repr__(self):
        return "FC Tied(input: {}, hidden: {})".format(self.input_size, self.hidden_size)


class DurationRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)
    
    def __repr__(self):
        return "DurationRegression(input: {})".format(self.input_size)


class Model(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, layer_num, w_drop, dropout_i, dropout_l, dropout_o,
                 dropout_e, winit, lstm_type):
        super().__init__()
        self.embed = Embed(vocab_size, embed_size, dropout_e, winit)
        self.rnns = [WeightDropLSTMCustom(embed_size if i == 0 else hidden_size, embed_size if i == layer_num-1 else hidden_size, w_drop) if lstm_type == "custom"
                     else WeightDropLSTM(embed_size if i == 0 else hidden_size, embed_size if i == layer_num-1 else hidden_size, w_drop) for i in range(layer_num)]
        self.rnns = nn.ModuleList(self.rnns)
        self.fc = FCTied(embed_size, vocab_size, self.embed.W)
        self.duration_regression = DurationRegression(embed_size)
        self.dropout = VariationalDropout()
        self.dropout_i = dropout_i
        self.dropout_l = dropout_l
        self.dropout_o = dropout_o
        self.lstm_type = lstm_type

    def forward(self, x, states):
        x = self.embed(x)
        x_m = self.dropout(x, self.dropout_i)
        for i, rnn in enumerate(self.rnns):
            x, states[i] = rnn(x_m, states[i])
            x_m = self.dropout(x, self.dropout_l if i != len(self.rnns)-1 else self.dropout_o)
        scores = self.fc(x_m)
        
        fix_durations = self.duration_regression(x.reshape(-1, x.shape[-1])).squeeze()
        
        if self.training:
            return scores, states, (x, x_m), fix_durations
        else:
            return scores, states
    
    def state_init(self, batch_size):
        dev = next(self.parameters()).device
        states = [(torch.zeros(batch_size, layer.hidden_size, device = dev), torch.zeros(batch_size, layer.hidden_size, device = dev)) if self.lstm_type == "custom" 
                  else (torch.zeros(1, batch_size, layer.hidden_size, device = dev), torch.zeros(1, batch_size, layer.hidden_size, device = dev)) for layer in self.rnns]
        return states
            
    @staticmethod
    def detach(states):
        return [(h.detach(), c.detach()) for (h, c) in states]
    