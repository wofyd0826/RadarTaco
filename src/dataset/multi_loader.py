"""Probabilistic multi-dataset loader.

Per step, picks one of the wrapped DataLoaders by `probs` and yields its
next batch (auto-restart on StopIteration).

Epoch length is anchored to a single "primary" loader (typically the real
nuScenes loader at index 0) so that sim-mixing does NOT inflate epoch
time. The number of steps per epoch is

    steps_per_epoch = round(len(primary) / probs[primary_idx])

i.e. the total step count needed for the primary loader to be fully
consumed in expectation. With ratio_real=0.5 and len(primary)=2344 this
gives 4688 steps: ~2344 primary batches + ~2344 sim batches per epoch.
With ratio_real=0.7 it gives ~3349 steps. Sim sources auto-restart if
they run out within an epoch.

This keeps validation cadence and lr-schedule arithmetic invariant to
the sim-mixing ratio — important for clean ablations.
"""
import logging
import random
from typing import Iterable, Iterator, List

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class MultiDatasetLoader:
    def __init__(
        self,
        loaders: Iterable[DataLoader],
        probs: Iterable[float],
        primary_idx: int = 0,
    ) -> None:
        self.loaders: List[DataLoader] = list(loaders)
        probs = list(probs)
        if len(self.loaders) != len(probs):
            raise ValueError("loaders and probs must have the same length")
        s = float(sum(probs))
        if s <= 0:
            raise ValueError("probs must sum to > 0")
        self.probs: List[float] = [p / s for p in probs]
        if not 0 <= primary_idx < len(self.loaders):
            raise ValueError(f"primary_idx {primary_idx} out of range")
        self.primary_idx = primary_idx
        self._rng = random.Random()
        self._steps = self._compute_steps_per_epoch()
        logger.info(
            f"MultiDatasetLoader: {len(self.loaders)} sources, "
            f"primary_idx={self.primary_idx} "
            f"(len={len(self.loaders[self.primary_idx])}, "
            f"prob={self.probs[self.primary_idx]:.2f}) "
            f"→ steps/epoch = {self._steps}"
        )

    def _compute_steps_per_epoch(self) -> int:
        primary_len = len(self.loaders[self.primary_idx])
        primary_prob = self.probs[self.primary_idx]
        if primary_prob <= 0:
            # Pathological: primary never picked. Fall back to summed length.
            return sum(len(l) for l in self.loaders)
        # Total step count so the primary is fully consumed in expectation.
        return int(round(primary_len / primary_prob))

    def __len__(self) -> int:
        return self._steps

    def __iter__(self) -> Iterator[dict]:
        iters = [iter(l) for l in self.loaders]
        for _ in range(self._steps):
            i = self._rng.choices(range(len(self.loaders)), weights=self.probs, k=1)[0]
            try:
                yield next(iters[i])
            except StopIteration:
                iters[i] = iter(self.loaders[i])
                yield next(iters[i])
