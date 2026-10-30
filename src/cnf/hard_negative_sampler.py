"""Deterministic, class-balanced sampling for CNF rule-triggered negatives.

The sampler changes training exposure only.  It derives hard-negative status
from the training pair's label-blind symbolic cache and never reads validation
or test labels.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Sampler


class EpochBalancedHardNegativeSampler(Sampler):
    """Keep the original class balance while controlling hard negatives."""

    def __init__(self, positives, hard_negatives, easy_negatives, hard_share, seed):
        self.positives = tuple(positives)
        self.hard_negatives = tuple(hard_negatives)
        self.easy_negatives = tuple(easy_negatives)
        self.hard_share = float(hard_share)
        self.seed = int(seed)
        self.epoch = 0
        if not 0.0 <= self.hard_share <= 1.0:
            raise ValueError("hard_negative_share must lie in [0, 1]")
        if not self.positives or not self.hard_negatives or not self.easy_negatives:
            raise ValueError("positive, hard-negative, and easy-negative pools must be non-empty")
        if len(self.positives) != len(self.hard_negatives) + len(self.easy_negatives):
            raise ValueError("formal CNF sampler requires the frozen 1:1 class balance")

    @staticmethod
    def _draw(pool, count, generator):
        """Draw evenly across a pool; duplicate only when the target exceeds it."""
        pool = torch.tensor(pool, dtype=torch.long)
        selected = []
        remaining = count
        while remaining:
            order = torch.randperm(len(pool), generator=generator)
            take = min(remaining, len(pool))
            selected.append(pool[order[:take]])
            remaining -= take
        return torch.cat(selected) if selected else torch.empty(0, dtype=torch.long)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + 1_000_003 * self.epoch)
        self.epoch += 1
        negative_count = len(self.hard_negatives) + len(self.easy_negatives)
        hard_count = round(negative_count * self.hard_share)
        easy_count = negative_count - hard_count
        chosen = torch.cat((
            self._draw(self.positives, len(self.positives), generator),
            self._draw(self.hard_negatives, hard_count, generator),
            self._draw(self.easy_negatives, easy_count, generator),
        ))
        order = torch.randperm(len(chosen), generator=generator)
        return iter(chosen[order].tolist())

    def __len__(self):
        return len(self.positives) + len(self.hard_negatives) + len(self.easy_negatives)


def build_training_sampler(data_root, symbolic_cache_name, hard_share, seed,
                           continuous_threshold=0.5):
    """Build an index sampler from train-only labels and label-blind features."""
    root = Path(data_root)
    feature_path = root / symbolic_cache_name
    features = {}
    with feature_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            features[(row["conjecture"], row["premise"])] = (
                float(row["trigger"]), float(row["weighted_similarity_score"]))

    positives, hard_negatives, easy_negatives = [], [], []
    raw_path = root / "train" / "raw" / "train.json"
    with raw_path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            conjecture, premise, label = json.loads(line)
            key = (conjecture, premise)
            if key not in features:
                raise KeyError(f"symbolic cache misses training pair {key}")
            if int(label) == 1:
                positives.append(index)
            elif int(label) == 0:
                trigger, continuous = features[key]
                target = (hard_negatives if trigger >= 0.5 or
                          continuous >= continuous_threshold else easy_negatives)
                target.append(index)
            else:
                raise ValueError(f"non-binary training label at row {index}")

    sampler = EpochBalancedHardNegativeSampler(
        positives, hard_negatives, easy_negatives, hard_share, seed)
    metadata = {
        "mode": "class_balanced_rule_hard_negative",
        "hard_negative_definition": (
            f"training label=0 and (trigger>=0.5 or weighted_similarity_score>={continuous_threshold:g})"),
        "hard_negative_share": float(hard_share),
        "positive_count": len(positives),
        "negative_count": len(hard_negatives) + len(easy_negatives),
        "hard_negative_count": len(hard_negatives),
        "easy_negative_count": len(easy_negatives),
        "natural_hard_negative_share": len(hard_negatives) /
        (len(hard_negatives) + len(easy_negatives)),
        "epoch_samples": len(sampler),
        "seed": int(seed),
        "uses_validation_or_test_labels": False,
    }
    return sampler, metadata
