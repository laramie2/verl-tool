import logging
import math
import re
from collections import defaultdict
from collections.abc import Sized
from typing import Any

import numpy as np

from verl.experimental.dataset.sampler import AbstractSampler

logger = logging.getLogger(__name__)


class StratifiedSourceSampler(AbstractSampler):
    """Yield sample indices in balanced batch-sized blocks.

    The sampler is intentionally configured through ``data.sampler`` so it can be
    enabled without changing trainer code. It supports either labels read from a
    dataset field, or explicit row ranges for already-concatenated datasets.
    """

    def __init__(self, data_source: Sized, data_config: Any):
        self.data_source = data_source
        self.data_config = data_config
        self.sampler_config = data_config.sampler
        self.seed = int(self.sampler_config.get("seed", data_config.get("seed", 1)))
        self.batch_size = int(data_config.get("gen_batch_size", data_config.train_batch_size))
        self.field = self.sampler_config.get("field", data_config.get("reward_fn_key", "data_source"))
        self.match_mode = self.sampler_config.get("match_mode", "exact")
        self.shuffle_within_batch = bool(self.sampler_config.get("shuffle_within_batch", True))
        self.drop_remainder = bool(self.sampler_config.get("drop_remainder", True))
        self.reshuffle_each_epoch = bool(self.sampler_config.get("reshuffle_each_epoch", True))
        self._iteration = 0

        self.buckets = self._build_buckets()
        self.labels = list(self.buckets)
        if len(self.labels) < 2:
            raise ValueError(f"StratifiedSourceSampler requires at least two strata, got labels={self.labels}")

        self.per_label_counts = self._build_per_label_counts()
        self.num_batches = self._compute_num_batches()
        if self.num_batches <= 0:
            raise ValueError(
                "No full stratified batches can be built. "
                f"batch_size={self.batch_size}, per_label_counts={self.per_label_counts}, "
                f"bucket_sizes={self._bucket_sizes()}"
            )

        logger.info(
            "Using StratifiedSourceSampler: labels=%s, per_label_counts=%s, num_batches=%s, bucket_sizes=%s",
            self.labels,
            self.per_label_counts,
            self.num_batches,
            self._bucket_sizes(),
        )

    def __iter__(self):
        seed = self.seed + self._iteration if self.reshuffle_each_epoch else self.seed
        if self.reshuffle_each_epoch:
            self._iteration += 1
        return iter(self._build_epoch_indices(seed))

    def __len__(self) -> int:
        return self.num_batches * self.batch_size

    def _build_buckets(self) -> dict[str, list[int]]:
        if self.sampler_config.get("label_ranges", None):
            return self._build_range_buckets()
        return self._build_field_buckets()

    def _build_range_buckets(self) -> dict[str, list[int]]:
        buckets = {}
        dataset_len = len(self.data_source)
        for item in self.sampler_config.label_ranges:
            label = str(item["label"])
            start = int(item.get("start", 0))
            end = int(item.get("end", dataset_len))
            if start < 0 or end > dataset_len or start >= end:
                raise ValueError(f"Invalid label range for {label}: start={start}, end={end}, len={dataset_len}")
            buckets[label] = list(range(start, end))
        return buckets

    def _build_field_buckets(self) -> dict[str, list[int]]:
        labels = self._to_list(self.sampler_config.get("labels", None))
        dataframe = getattr(self.data_source, "dataframe", None)
        if dataframe is None:
            raise ValueError("StratifiedSourceSampler needs data_source.dataframe when label_ranges is not configured")

        values = [self._get_nested_value(row, self.field) for row in dataframe]
        if labels is None:
            labels = sorted({str(value) for value in values})

        buckets = {str(label): [] for label in labels}
        unmatched = 0
        for idx, value in enumerate(values):
            matched_label = self._match_label(str(value), labels)
            if matched_label is None:
                unmatched += 1
                continue
            buckets[str(matched_label)].append(idx)

        empty = [label for label, indices in buckets.items() if not indices]
        if empty:
            raise ValueError(
                f"No samples found for labels={empty} using field={self.field!r}, match_mode={self.match_mode!r}"
            )
        if unmatched:
            logger.warning("StratifiedSourceSampler ignored %s samples that did not match configured labels", unmatched)
        return buckets

    def _build_per_label_counts(self) -> dict[str, int]:
        counts = self.sampler_config.get("per_label_counts", None)
        if counts is not None:
            counts = dict(counts)
            per_label_counts = {label: int(counts[label]) for label in self.labels}
        else:
            ratios = self.sampler_config.get("ratios", None)
            if ratios is None:
                if self.batch_size % len(self.labels) != 0:
                    raise ValueError(
                        f"batch_size={self.batch_size} must be divisible by strata={len(self.labels)} "
                        "when data.sampler.ratios/per_label_counts is not set"
                    )
                per_label_counts = {label: self.batch_size // len(self.labels) for label in self.labels}
            else:
                per_label_counts = self._ratios_to_counts(ratios)

        total = sum(per_label_counts.values())
        if total != self.batch_size:
            raise ValueError(f"per-label counts must sum to batch_size={self.batch_size}, got {per_label_counts}")
        if any(count <= 0 for count in per_label_counts.values()):
            raise ValueError(f"per-label counts must be positive, got {per_label_counts}")
        return per_label_counts

    def _ratios_to_counts(self, ratios: Any) -> dict[str, int]:
        if self._is_list_like(ratios):
            if len(ratios) != len(self.labels):
                raise ValueError(f"ratios length {len(ratios)} must match labels length {len(self.labels)}")
            ratio_map = {label: float(ratio) for label, ratio in zip(self.labels, ratios, strict=True)}
        else:
            ratio_map = {label: float(ratios[label]) for label in self.labels}

        total_ratio = sum(ratio_map.values())
        raw_counts = {label: ratio_map[label] / total_ratio * self.batch_size for label in self.labels}
        counts = {label: int(math.floor(raw_counts[label])) for label in self.labels}
        remaining = self.batch_size - sum(counts.values())

        by_remainder = sorted(self.labels, key=lambda label: raw_counts[label] - counts[label], reverse=True)
        for label in by_remainder[:remaining]:
            counts[label] += 1
        return counts

    def _compute_num_batches(self) -> int:
        if self.drop_remainder:
            return min(len(self.buckets[label]) // self.per_label_counts[label] for label in self.labels)
        return min(math.ceil(len(self.buckets[label]) / self.per_label_counts[label]) for label in self.labels)

    def _build_epoch_indices(self, seed: int) -> list[int]:
        rng = np.random.default_rng(seed)
        buckets = {label: np.array(indices, dtype=np.int64) for label, indices in self.buckets.items()}
        for indices in buckets.values():
            rng.shuffle(indices)

        ordered = []
        offsets = defaultdict(int)
        for _ in range(self.num_batches):
            batch = []
            for label in self.labels:
                count = self.per_label_counts[label]
                start = offsets[label]
                end = start + count
                if end <= len(buckets[label]):
                    batch.extend(buckets[label][start:end].tolist())
                else:
                    tail = buckets[label][start:].tolist()
                    refill = rng.choice(buckets[label], size=end - len(buckets[label]), replace=True).tolist()
                    batch.extend(tail + refill)
                offsets[label] = end
            if self.shuffle_within_batch:
                rng.shuffle(batch)
            ordered.extend(batch)
        return ordered

    def _bucket_sizes(self) -> dict[str, int]:
        return {label: len(indices) for label, indices in self.buckets.items()}

    def _match_label(self, value: str, labels: list[Any]) -> str | None:
        for label in labels:
            label = str(label)
            if self.match_mode == "exact" and value == label:
                return label
            if self.match_mode == "contains" and label in value:
                return label
            if self.match_mode == "regex" and re.search(label, value):
                return label
        return None

    @staticmethod
    def _to_list(value: Any) -> list[Any] | None:
        if value is None:
            return None
        if StratifiedSourceSampler._is_list_like(value):
            return list(value)
        return [value]

    @staticmethod
    def _get_nested_value(row: dict[str, Any], field: str) -> Any:
        value = row
        for part in field.split("."):
            if isinstance(value, dict):
                value = value[part]
            else:
                value = getattr(value, part)
        return value

    @staticmethod
    def _is_list_like(value: Any) -> bool:
        if isinstance(value, (str, bytes, dict)) or value is None:
            return False
        return isinstance(value, (list, tuple)) or value.__class__.__name__ == "ListConfig"