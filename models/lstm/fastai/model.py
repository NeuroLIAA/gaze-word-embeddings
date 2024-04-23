import numpy as np
import torch.cuda

from fastai.basic_data import DataBunch
from fastai.datasets import Config, shutil
from fastai.vision import *
from fastai.text import (
    TextList, language_model_learner, AWD_LSTM, load_learner, LanguageLearner,
    NumericalizeProcessor,
)

from eval_model import evaluate
from preprocess import FileCustomTokenizerMaj
from pathlib import Path

if torch.cuda.is_available():
    torch.cuda.set_device('cuda:1')
    print("running on cuda")

def train(folder, bs, n):
    model_folder = Path('./data/models')
    if not (model_folder / 'model_vocab.pkl').exists():
        model_folder.mkdir(exist_ok=True)
        data: DataBunch = (
            TextList
                .from_folder(folder, processor=[FileCustomTokenizerMaj(), NumericalizeProcessor()])
                .split_by_folder()
                .label_for_lm()
                .databunch(bs=bs, num_workers=1)
        )
        data.save('models/data.databunch')

        print('Data bunch crated')
        "añadir to_fp16 a la linea de abajo cuando se entrene con gpu"
        learn: LanguageLearner = language_model_learner(data, AWD_LSTM, drop_mult=0.5, pretrained=False, 
                                                        callback_fns=[partial(callbacks.CSVLogger, filename="fastai_results")])
        lr = 2e-2
        learn.unfreeze()
        learn.fit_one_cycle(n, lr / n, moms=(0.8, 0.7))
        learn.to_fp32()
        learn.save('fastai_model')
        learn.data.vocab.save('data/models/model_vocab.pkl')
        learn.export(file=f'models/export.pkl')
    return model_folder


if __name__ == '__main__':
    train(
        folder= Path('./data'),
        bs=128,
        n=1
    )
