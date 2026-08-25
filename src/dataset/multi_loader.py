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

import torch
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
        primary_prob = self.probs[self.primary_idx]
        if primary_prob <= 0:
            # Configured primary never picked (e.g. ratio_real=0 for a
            # sim-only ablation). Re-anchor to whichever loader has the
            # highest probability so epoch length tracks that source.
            self.primary_idx = max(range(len(self.probs)), key=lambda i: self.probs[i])
            primary_prob = self.probs[self.primary_idx]
        primary_len = len(self.loaders[self.primary_idx])
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


class MixedBatchLoader:
    """Per-step yields a SINGLE batch composed of fixed quotas from each
    source loader. Unlike `MultiDatasetLoader` (which yields a whole batch
    from one source per step), this concatenates `quotas[i]` samples from
    source loader `i` into one batch.

    Use case: have nuScenes and one-or-more sim sources contribute their
    own slice of every batch — keeps BN/grad statistics balanced across
    domains within each step instead of alternating.

    Constraint: all source loaders must produce tensors of the same H/W
    (the resulting batch is `torch.cat`-ed along dim 0). Caller is
    responsible for matching resolutions (e.g. force sim resize to nuScenes
    native).

    Each source loader's `batch_size` must be set to its `quotas[i]`.
    Epoch length is anchored to the primary loader fully consuming its
    dataset:
        steps_per_epoch = floor(len(primary_dataset) / quotas[primary_idx])
    """

    def __init__(
        self,
        source_loaders: Iterable[DataLoader],
        quotas: Iterable[int],
        primary_idx: int = 0,
    ) -> None:
        self.loaders: List[DataLoader] = list(source_loaders)
        self.quotas: List[int] = [int(q) for q in quotas]
        if len(self.loaders) != len(self.quotas):
            raise ValueError("source_loaders and quotas length mismatch")
        if any(q < 0 for q in self.quotas):
            raise ValueError(f"quotas must be non-negative, got {self.quotas}")
        if sum(self.quotas) <= 0:
            raise ValueError("at least one quota must be > 0")
        # Re-anchor primary if its quota is 0
        if self.quotas[primary_idx] == 0:
            primary_idx = max(range(len(self.quotas)), key=lambda i: self.quotas[i])
        self.primary_idx = primary_idx
        primary_dataset_len = len(self.loaders[self.primary_idx].dataset)
        primary_quota = self.quotas[self.primary_idx]
        self._steps = primary_dataset_len // primary_quota
        self._batch_size = sum(self.quotas)
        logger.info(
            f"MixedBatchLoader: {len(self.loaders)} sources, "
            f"quotas={self.quotas} (total batch={self._batch_size}), "
            f"primary_idx={self.primary_idx} "
            f"(dataset_len={primary_dataset_len}) "
            f"→ steps/epoch = {self._steps}"
        )

    def __len__(self) -> int:
        return self._steps

    @staticmethod
    def _concat_batches(sub_batches: List[dict]) -> dict:
        """Concatenate per-source mini-batches into one full batch.

        - Tensors → torch.cat along dim 0.
        - Lists (e.g. sample_id collated as list-of-strings) → list concat.
        - Anything else → first non-None value (assume per-batch metadata).
        """
        keys = list(sub_batches[0].keys())
        out = {}
        for k in keys:
            vals = [b[k] for b in sub_batches if k in b]
            if not vals:
                continue
            if torch.is_tensor(vals[0]):
                out[k] = torch.cat(vals, dim=0)
            elif isinstance(vals[0], list):
                merged = []
                for v in vals:
                    merged.extend(v)
                out[k] = merged
            else:
                out[k] = vals[0]
        return out

    def __iter__(self) -> Iterator[dict]:
        iters = [iter(l) for l in self.loaders]
        for _ in range(self._steps):
            sub_batches = []
            for i in range(len(self.loaders)):
                if self.quotas[i] == 0:
                    continue
                try:
                    sub_batches.append(next(iters[i]))
                except StopIteration:
                    # auto-restart secondary sources mid-epoch
                    iters[i] = iter(self.loaders[i])
                    sub_batches.append(next(iters[i]))
            yield self._concat_batches(sub_batches)
