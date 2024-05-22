import torch
import torch.optim as optim
from tqdm import tqdm
from scripts.w2v_fix import SkipGram
from scripts.plot import plot_loss
from scripts.data_handling import get_dataloader_and_vocab

class Word2vecTorch:
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
        dataloader, vocab = get_dataloader_and_vocab(self.corpora, self.min_count, self.negative_samples, self.downsample_factor,
                                                    self.window_size, self.batch_size, self.train_fix)
        skip_gram = SkipGram(len(vocab), self.vector_size)
        if device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            skip_gram.cuda()
        else:
            device = torch.device('cpu')

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
                    pos_u = batch[0].to(device)
                    pos_v = batch[1].to(device)
                    neg_v = batch[2].to(device)
                    fix_v = batch[3].to(device)

                    update_regressor = self.train_fix and fix_v.sum() > 0
                    opt_sparse.zero_grad(), opt_dense.zero_grad()
                    loss, fix_dur = skip_gram.forward(pos_u, pos_v, neg_v, self.train_fix)
                    loss_sg.append(loss.item())
                    if update_regressor:
                        fix_loss = torch.nn.L1Loss()(fix_dur, fix_v)
                        scale_factor = loss_sg[-1] / fix_loss.item()
                        fix_loss *= scale_factor
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