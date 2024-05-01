from fastai.basics import *
from fastai.callback.all import *
from fastai.text.all import *

path = untar_data(URLs.WIKITEXT_TINY)

df_train = pd.read_csv("./data/train/train.txt", header=None)
df_valid = pd.read_csv("./data/valid/valid.txt", header=None)
df_all = pd.concat([df_train, df_valid])

splits = [list(range_of(df_train)), list(range(len(df_train), len(df_all)))]
tfms = [attrgetter("text"), Tokenizer.from_df(0), Numericalize()]
dsets = Datasets(df_all, [tfms], splits=splits, dl_type=LMDataLoader)

bs,sl = 104,72
dls = dsets.dataloaders(bs=bs, seq_len=sl)

config = awd_lstm_lm_config.copy()
config.update({'input_p': 0.6, 'output_p': 0.4, 'weight_p': 0.5, 'embed_p': 0.1, 'hidden_p': 0.2})
model = get_language_model(AWD_LSTM, len(dls.vocab), config=config)

opt_func = partial(Adam, wd=0.1, eps=1e-7)
cbs = [MixedPrecision(), GradientClip(0.1), CSVLogger(fname="./data/fastai_results.csv")] + rnn_cbs(alpha=2, beta=1)

learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), opt_func=opt_func, cbs=cbs, metrics=[accuracy, Perplexity()])

learn.fit_one_cycle(300, 5e-3, moms=(0.8,0.7,0.8), div=10)