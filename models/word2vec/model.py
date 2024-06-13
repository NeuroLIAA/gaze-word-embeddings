import torch
import torch.optim as optim
from torch import nn as nn
from torch.nn import init, functional
from tqdm import tqdm
from scripts.plot import plot_loss
from scripts.data_handling import get_dataloader_and_vocab


class Word2Vec:
    def __init__(self, corpora, vector_size, window_size, min_count, negative_samples, downsample_factor, epochs, lr,
                 batch_size, train_fix, device, model_name, save_path):
        self.corpora = corpora
        self.vector_size = vector_size
        self.window_size = window_size
        self.min_count = min_count
        self.negative_samples = negative_samples
        self.downsample_factor = downsample_factor
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.train_fix = train_fix
        self.device = device
        self.model_name = model_name
        self.save_path = save_path

    def train(self):
        dataloader, vocab = get_dataloader_and_vocab(self.corpora, self.min_count, self.negative_samples,
                                                     self.downsample_factor, self.window_size, self.batch_size,
                                                     self.train_fix)
        skip_gram = SkipGram(len(vocab), self.vector_size)
        if self.device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            skip_gram.cuda()
        else:
            self.device = torch.device('cpu')

        loss_sg, loss_fix = [], []
        for epoch in range(self.epochs):
            print(f'\nEpoch: {epoch + 1}')
            sparse_params = list(skip_gram.parameters())[:2]
            dense_params = list(skip_gram.parameters())[2:]
            opt_sparse = optim.SparseAdam(sparse_params, lr=self.lr)
            opt_dense = optim.AdamW(dense_params, lr=self.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_sparse, len(dataloader))
            for batch in tqdm(dataloader):
                if len(batch[0]) > 1:
                    pos_u = batch[0].to(self.device)
                    pos_v = batch[1].to(self.device)
                    neg_v = batch[2].to(self.device)
                    fix_v = batch[3].to(self.device)

                    update_regressor = self.train_fix and fix_v.sum() > 0
                    opt_sparse.zero_grad(), opt_dense.zero_grad()
                    loss, fix_dur = skip_gram.forward(pos_u, pos_v, neg_v, self.train_fix)
                    loss_sg.append(loss.item())
                    if update_regressor:
                        fix_loss = torch.nn.functional.mse_loss(torch.argmax(fix_dur, dim=1).to(torch.float), fix_v)
                        loss += fix_loss
                        loss_fix.append(fix_loss.item())
                    else:
                        loss_fix.append(loss_fix[-1] if loss_fix else 0)
                    loss.backward()
                    opt_sparse.step()
                    if update_regressor:
                        opt_dense.step()
                    scheduler.step()

        self.save_path.mkdir(exist_ok=True, parents=True)
        skip_gram.save_embedding_vocab(vocab, str(self.save_path / f'{self.model_name}.vec'))
        plot_loss(loss_sg, loss_fix, self.model_name, self.save_path)


class SkipGram(nn.Module):

    def __init__(self, emb_size, emb_dimension, num_classes=6):
        super(SkipGram, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.duration_regression = nn.Linear(emb_dimension, num_classes)

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

    def save_embedding_vocab(self, vocab, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(vocab), self.emb_dimension))
            for wid, w in enumerate(vocab.get_itos()):
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))
