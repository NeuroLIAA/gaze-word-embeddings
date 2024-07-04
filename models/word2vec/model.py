import torch
import torch.optim as optim
from torch import nn as nn
from torch.nn import init, functional
from tqdm import tqdm
from scripts.plot import plot_loss
from scripts.data_handling import get_dataloader_and_vocab


class Word2Vec:
    def __init__(self, corpora, vector_size, window_size, min_count, negative_samples, downsample_factor, epochs, lr,
                 min_lr, batch_size, train_fix, stimuli_path, device, model_name, pretrained_path, save_path):
        self.corpora = corpora
        self.vector_size = vector_size
        self.window_size = window_size
        self.min_count = min_count
        self.negative_samples = negative_samples
        self.downsample_factor = downsample_factor
        self.epochs = epochs
        self.lr = lr
        self.min_lr = min_lr
        self.batch_size = batch_size
        self.train_fix = train_fix
        self.device = device
        self.model_name = model_name
        self.stimuli_path = stimuli_path
        self.pretrained_path = pretrained_path
        self.save_path = save_path

    def train(self):
        self.save_path.mkdir(exist_ok=True, parents=True)
        dataloader, vocab = get_dataloader_and_vocab(self.corpora, self.min_count, self.negative_samples,
                                                     self.downsample_factor, self.window_size, self.batch_size,
                                                     self.train_fix, self.stimuli_path, self.pretrained_path,
                                                     self.save_path)
        device = torch.device('cuda' if torch.cuda.is_available() and self.device == 'cuda' else 'cpu')
        skip_gram = SkipGram(len(vocab), self.vector_size, self.lr, device)
        if self.pretrained_path:
            skip_gram.load_checkpoint(self.pretrained_path, device)

        loss_sg, loss_fix = [], []
        for epoch in range(self.epochs):
            print(f'\nEpoch: {epoch + 1}')
            for batch in tqdm(dataloader):
                if len(batch[0]) > 1:
                    pos_u = batch[0].to(device)
                    pos_v = batch[1].to(device)
                    neg_v = batch[2].to(device)
                    fix_v = batch[3].to(device)

                    update_regressor = self.train_fix and fix_v.sum() > 0
                    skip_gram.optimizers['embeddings'].zero_grad()
                    skip_gram.optimizers['fix_duration'].zero_grad()
                    loss, fix_dur = skip_gram.forward(pos_u, pos_v, neg_v, self.train_fix)
                    loss_sg.append(loss.item())
                    if update_regressor:
                        fix_loss = torch.nn.functional.mse_loss(torch.argmax(fix_dur, dim=1).to(torch.float), fix_v)
                        loss += fix_loss
                        loss_fix.append(fix_loss.item())
                    else:
                        loss_fix.append(loss_fix[-1] if loss_fix else 0)
                    loss.backward()
                    skip_gram.optimizers['embeddings'].step()
                    if update_regressor:
                        skip_gram.optimizers['fix_duration'].step()
            skip_gram.save_checkpoint(self.save_path / f'{self.model_name}.pt', epoch, loss_sg, loss_fix)

        skip_gram.save_embedding_vocab(vocab, str(self.save_path / f'{self.model_name}.vec'))
        plot_loss(loss_sg, loss_fix, self.model_name, self.save_path)


class SkipGram(nn.Module):

    def __init__(self, vocab_size, emb_dimension, lr, device, num_classes=6):
        super(SkipGram, self).__init__()
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)
        self.duration_regression = nn.Linear(emb_dimension, num_classes)
        self.optimizers = self.init_optimizers(lr)
        self.to(device)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v, predict_fix):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -functional.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(functional.logsigmoid(-neg_score), dim=1)

        duration = self.duration_regression(emb_v).squeeze() if predict_fix == 'output' else self.duration_regression(
            emb_u).squeeze()

        return torch.mean(score + neg_score), duration

    def init_optimizers(self, lr):
        optimizers = {'embeddings': optim.SparseAdam(list(self.parameters())[:2], lr=lr),
                        'fix_duration': optim.AdamW(list(self.parameters())[2:], lr=lr)}
        return optimizers

    def save_embedding_vocab(self, vocab, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(vocab), self.emb_dimension))
            for wid, w in enumerate(vocab.get_itos()):
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))

    def save_checkpoint(self, file_name, epoch, loss_sg, loss_fix):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': [self.optimizers[opt].state_dict() for opt in self.optimizers],
            'loss_sg': loss_sg,
            'loss_fix': loss_fix
        }, file_name)

    def load_checkpoint(self, file_name, device):
        checkpoint = torch.load(file_name, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        for opt, state in zip(self.optimizers, checkpoint['optimizer_state_dict']):
            self.optimizers[opt].load_state_dict(state)
