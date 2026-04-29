"""Probabilistic multi-dataset loader.

Per step, picks one of the wrapped DataLoaders by `probs` and yields its
next batch (auto-restart on StopIteration). `__len__` is the sum of wrapped
loader lengths, so an "epoch" iterates that many steps with mixing.
"""
import random
from typing import Iterable, Iterator, List

from torch.utils.data import DataLoader


class MultiDatasetLoader:
    def __init__(self, loaders: Iterable[DataLoader], probs: Iterable[float]) -> None:
        self.loaders: List[DataLoader] = list(loaders)
        probs = list(probs)
        if len(self.loaders) != len(probs):
            raise ValueError("loaders and probs must have the same length")
        s = float(sum(probs))
        if s <= 0:
            raise ValueError("probs must sum to > 0")
        self.probs = [p / s for p in probs]
        self._rng = random.Random()

    def __len__(self) -> int:
        return sum(len(l) for l in self.loaders)

    def __iter__(self) -> Iterator[dict]:
        iters = [iter(l) for l in self.loaders]
        for _ in range(len(self)):
            i = self._rng.choices(range(len(self.loaders)), weights=self.probs, k=1)[0]
            try:
                yield next(iters[i])
            except StopIteration:
                iters[i] = iter(self.loaders[i])
                yield next(iters[i])
