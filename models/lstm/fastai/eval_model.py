import re

import torch
from fastai.text import load_learner, LanguageLearner, BOS, UNK

def is_valid_word(word: str):
    return not word.startswith("xx") and re.match(r"^\w+$", word) is not None


def evaluate(ll: LanguageLearner, text: str):
    ll.model.reset()
    xtokens, yb = ll.data.one_item(text)
    xb = torch.tensor([[ll.data.vocab.stoi[BOS]]])

    probs = []
    is_a_word = torch.tensor([is_valid_word(w) for w in ll.data.vocab.itos]).float()

    for token in xtokens[0][1:]:
        word = ll.data.vocab.itos[token]
        res = ll.pred_batch(batch=(xb, yb))[0][-1]
        res *= is_a_word
        prob = res[token] / res.sum()
        if is_valid_word(word) or word == UNK:
            probs.append((word, prob.item()))
        xb = xb.new_tensor([token])[None]

    return probs
