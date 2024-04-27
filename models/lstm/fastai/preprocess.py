import re
from pathlib import Path
from typing import overload

from fastai.data_block import PreProcessor
from fastai.text import BOS, EOS, TK_UP, TK_MAJ


class FileCustomTokenizer(PreProcessor):

    def __init__(self, ds=None):
        super().__init__(ds)
        self.rr = re.compile(r'(?<=\w)[^\w\s](?=\w)')
        self.rgx = re.compile(r'(\w+|[^\w\s])')
        self.lws = []
        self.dd = {}

    def process(self, ds):
        items = []
        for i, item in enumerate(ds):
            if i % 1_000 == 0:
                print(f"\r{i} of {len(ds)}: {100 * i / len(ds):0.3f}%", end="")
            items.append(self.process_one(item))
        ds.items = items
        return ds

    def process_one(self, item):
        if isinstance(item, int):
            return [item]
        if isinstance(item, Path):
            with item.open('r') as f:
                item = f.read()

        item = self.rr.sub('', item)

        return CompressedList(
            dd=self.dd,
            lws=self.lws,
            ws=['xxbos'] + self.rgx.findall(item) + ['xxeos']
        )


class FileCustomTokenizerMaj(FileCustomTokenizer):

    def process_one(self, item):
        if isinstance(item, int):
            return [item]
        if isinstance(item, Path):
            with item.open('r') as f:
                item = f.read()

        item = self.rr.sub('', item)

        return CompressedList(
            dd=self.dd,
            lws=self.lws,
            ws=[BOS, *self.transform_tokens(item), EOS]
        )

    def transform_tokens(self, item: str):
        for token in self.rgx.findall(item):
            token: str
            if token.istitle():
                yield TK_MAJ
                yield token.lower()
            elif token.isupper():
                yield TK_UP
                yield token.lower()
            else:
                yield token


class CompressedList:

    def _actualize_dd(self, dd: dict, lws: list, ws: list):
        for w in ws:
            if w not in dd:
                dd[w] = len(dd)
                lws.append(w)

    def __init__(self, dd: dict, lws: list, ws: list):
        self._actualize_dd(dd, lws, ws)
        self._items = [dd[w] for w in ws]
        self._lws = lws

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, i: int):
        return self._lws[self._items[i]]

    @overload
    def __getitem__(self, s: slice):
        return [self._lws[i] for i in self._items[s]]

    def __getitem__(self, i: int):
        return self._lws[self._items[i]]

    def __iter__(self):
        for item in self._items:
            yield self._lws[item]
