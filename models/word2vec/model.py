import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from torch.nn import init, functional
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import spearmanr
from scripts.data_handling import get_dataloader_and_vocab
from scripts.plot import plot_loss


class W2VTrainer:
    def __init__(self, corpora, vector_size, window_size, min_count, negative_samples, downsample_factor, epochs, lr,
                 min_lr, batch_size, gaze_table, stimuli_path, device, model_name, model_type,
                 pretrained_path, save_path):
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
        self.gaze_table = gaze_table
        self.device = device
        self.model_name = model_name
        self.model_type = model_type
        self.stimuli_path = stimuli_path
        self.pretrained_path = pretrained_path
        self.save_path = save_path

    def train(self):
        self.save_path.mkdir(exist_ok=True, parents=True)
        dataloader, vocab = get_dataloader_and_vocab(self.corpora, self.min_count, self.negative_samples,
                                                     self.downsample_factor, self.window_size, self.batch_size,
                                                     self.gaze_table, self.stimuli_path, self.pretrained_path,
                                                     self.model_type, self.save_path)
        device = torch.device('cuda' if torch.cuda.is_available() and self.device == 'cuda' else 'cpu')
        n_gaze_features = len(self.gaze_table.columns)
        model = Word2Vec(len(vocab), self.vector_size, self.lr, self.model_type, device, n_gaze_features)
        if self.pretrained_path:
            model.load_checkpoint(self.pretrained_path, device)

        fix_scheduler = optim.lr_scheduler.LinearLR(model.optimizers['fix_duration'], start_factor=1.0,
                                                    end_factor=(self.min_lr / self.lr), total_iters=self.epochs)
        scheduler = optim.lr_scheduler.LinearLR(model.optimizers['embeddings'], start_factor=1.0,
                                                end_factor=(self.min_lr / self.lr), total_iters=self.epochs)

        run_name = f'e{self.epochs}'
        writer = SummaryWriter(log_dir=self.save_path / 'logs' / run_name)
        loss_fix, loss_sg = [], []
        for epoch in range(self.epochs):
            print(f'\nEpoch: {epoch + 1}')
            fix_corrs = [[] for _ in range(n_gaze_features)]
            fix_pvalues = [[] for _ in range(n_gaze_features)]
            writer.add_scalar('lr/SG', model.optimizers['embeddings'].param_groups[0]['lr'], epoch)
            writer.add_scalar('lr/Fix', model.optimizers['fix_duration'].param_groups[0]['lr'], epoch)
            for n_step, batch in enumerate(tqdm(dataloader)):
                if len(batch[0]) > 1:
                    pos_u = batch[0].to(device)
                    pos_v = batch[1].to(device)
                    neg_v = batch[2].to(device)
                    fix_u = batch[3].to(device)
                    # fix_u has shape (batch_size, n_gaze_features)

                    fix_labels = fix_u[fix_u != -1].view(-1, n_gaze_features)
                    update_regressor = fix_labels.sum() > 0
                    model.optimizers['embeddings'].zero_grad(), model.optimizers['fix_duration'].zero_grad()
                    loss, fix_dur = model.forward(pos_u, pos_v, neg_v)
                    writer.add_scalar('Loss/SG', loss.item(), n_step)
                    loss_sg.append(loss.item())
                    fix_loss_value = 0.0
                    if update_regressor:
                        fix_loss = log_and_compute_loss(fix_dur, fix_labels, fix_corrs, fix_pvalues, n_gaze_features,
                                                        writer, n_step)
                        fix_loss_value = fix_loss.item()
                        loss += fix_loss
                    loss_fix.append(fix_loss_value)
                    loss.backward()
                    model.optimizers['embeddings'].step()
                    if update_regressor:
                        model.optimizers['fix_duration'].step()
            scheduler.step()
            fix_scheduler.step()
            model.save_checkpoint(self.save_path / f'{self.model_name}.pt', epoch)
            log_batch_corrs(self.gaze_table.columns, fix_corrs, fix_pvalues, n_gaze_features, writer, epoch)

        model.save_embedding_vocab(vocab, str(self.save_path / f'{self.model_name}.vec'))
        plot_loss(loss_sg, loss_fix, self.model_name, self.save_path, 'W2V')
        writer.flush()


class Word2Vec(nn.Module):

    def __init__(self, vocab_size, emb_dimension, lr, model_type, device, num_features=1):
        super(Word2Vec, self).__init__()
        self.emb_dimension = emb_dimension
        self.num_features = num_features
        self.u_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True, padding_idx=0)
        self.v_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True, padding_idx=0)
        self.duration_regression = nn.Linear(emb_dimension, num_features)
        self.optimizers = self.init_optimizers(lr)
        self.model_type = model_type
        self.device = device
        self.to(device)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, target_word, context_words, neg_words):
        emb_u = self.u_embeddings(target_word)
        emb_v = self.v_embeddings(context_words)
        emb_neg_v = self.v_embeddings(neg_words)

        if self.model_type == 'cbow':
            nonzero_mask = emb_u != 0
            predicted_fix = self.duration_regression(emb_u[nonzero_mask].view(-1, self.emb_dimension)).squeeze()
            emb_u = torch.sum(emb_u, dim=1) / torch.sum(nonzero_mask, dim=1)
        else:
            predicted_fix = self.duration_regression(emb_u).squeeze()
        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -functional.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(functional.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score), predicted_fix

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

    def save_checkpoint(self, file_name, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict()
        }, file_name)

    def load_checkpoint(self, checkpoint_path, device):
        checkpoint = next(checkpoint_path.glob(f'{self.model_type}*.pt'))
        checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)['model_state_dict']
        if self.num_features != checkpoint['duration_regression.weight'].shape[0]:
            print('Replacing duration regression layer from checkpoint')
            stdv = 1. / self.emb_dimension
            checkpoint['duration_regression.weight'] = (torch.FloatTensor(self.num_features, self.emb_dimension)
                                                        .uniform_(-stdv, stdv))
            checkpoint['duration_regression.bias'] = torch.FloatTensor(self.num_features).uniform_(-stdv, stdv)
        self.load_state_dict(checkpoint)


def calculate_class_weights(labels, num_classes):
    batch_classes = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=batch_classes, y=labels)
    full_class_weights = np.ones(num_classes, dtype=np.float32)

    for cls, weight in zip(batch_classes, class_weights):
        full_class_weights[cls] = weight

    return torch.tensor(full_class_weights, dtype=torch.float32)


def log_and_compute_loss(fix_dur, fix_labels, fix_corrs, fix_pvalues, n_gaze_features, writer, n_step):
    fix_loss = nn.functional.l1_loss(fix_dur, fix_labels)
    writer.add_scalar('Loss/Fix', fix_loss.item(), n_step)
    fix_preds = fix_dur.cpu().detach().numpy()
    fix_labels = fix_labels.cpu().detach().numpy()
    batch_correlations = [spearmanr(fix_preds[:, i], fix_labels[:, i], nan_policy='omit')
                          for i in range(fix_preds.shape[1])]
    for i in range(n_gaze_features):
        fix_corrs[i].append(batch_correlations[i].correlation)
        fix_pvalues[i].append(batch_correlations[i].pvalue)
    return fix_loss


def log_batch_corrs(gaze_features, fix_corrs, fix_pvalues, n_gaze_features, writer, epoch):
    if np.any(fix_corrs):
        for i in range(n_gaze_features):
            print(f'{gaze_features[i]} correlation: {np.nanmean(fix_corrs[i]):.3f} '
                  f'(+/- {np.nanstd(fix_corrs[i]):.3f}) | p-value: {np.nanmean(fix_pvalues[i]):.3f} '
                  f'(+/- {np.nanstd(fix_pvalues[i]):.3f})')
            writer.add_scalar(f'Correlation/{gaze_features[i]}', np.nanmean(fix_corrs[i]), epoch)
            writer.add_scalar(f'P-value/{gaze_features[i]}', np.nanmean(fix_pvalues[i]), epoch)
