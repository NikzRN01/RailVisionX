"""Utilities for selecting top-K frames by score.

This module is intentionally dependency-light (pure Python). It provides helpers
to:
- compute top-k indices from a sequence of scores
- pair frame indices with scores
- optionally enforce minimum spacing between selected frames
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import nlargest
from typing import Iterable, List, Optional, Sequence, Tuple, Union


Number = Union[int, float]


@dataclass(frozen=True)
class FrameScore:
	"""A score assigned to a frame."""

	frame_index: int
	score: float


def topk_indices(
	scores: Sequence[Number],
	k: int,
	*,
	largest: bool = True,
	stable: bool = True,
) -> List[int]:
	"""Return indices of the top-k scores.

	Args:
		scores: Sequence of numeric scores.
		k: Number of indices to return. If k <= 0, returns []. If k >= len(scores),
			returns all indices.
		largest: If True, select largest scores; otherwise select smallest.
		stable: If True, break ties by preferring lower indices.

	Returns:
		List of selected indices (not sorted by index; sorted by score rank).
	"""

	n = len(scores)
	if k <= 0 or n == 0:
		return []
	if k >= n:
		# Return all indices ordered by rank.
		k = n

	if stable:
		# Use (score, -index) for stable tie-breaking when picking largest.
		if largest:
			key = lambda i: (float(scores[i]), -i)
			return nlargest(k, range(n), key=key)
		else:
			# For smallest, invert score; keep tie-breaking stable.
			key = lambda i: (-float(scores[i]), -i)
			return nlargest(k, range(n), key=key)
	else:
		if largest:
			key = lambda i: float(scores[i])
			return nlargest(k, range(n), key=key)
		else:
			key = lambda i: -float(scores[i])
			return nlargest(k, range(n), key=key)


def topk_frames(
	scores: Sequence[Number],
	k: int,
	*,
	largest: bool = True,
	min_frame_distance: int = 0,
	stable: bool = True,
) -> List[FrameScore]:
	"""Select top-k frames by score.

	Args:
		scores: Sequence of scores aligned with frame index.
		k: Number of frames to select.
		largest: If True, select highest scores; otherwise lowest.
		min_frame_distance: If > 0, enforce a minimum absolute distance between
			selected frame indices. Greedy by score rank.
		stable: If True, break ties by preferring lower indices.

	Returns:
		List of FrameScore sorted by descending rank (best first).
	"""

	if min_frame_distance < 0:
		raise ValueError("min_frame_distance must be >= 0")

	ranked = topk_indices(scores, k=len(scores), largest=largest, stable=stable)
	if k <= 0:
		return []

	selected: List[FrameScore] = []
	taken: List[int] = []

	for idx in ranked:
		if min_frame_distance:
			# Reject if too close to any already selected frame.
			if any(abs(idx - t) < min_frame_distance for t in taken):
				continue
		selected.append(FrameScore(frame_index=int(idx), score=float(scores[idx])))
		taken.append(int(idx))
		if len(selected) >= k:
			break

	return selected


def topk_from_pairs(
	frame_score_pairs: Iterable[Tuple[int, Number]],
	k: int,
	*,
	largest: bool = True,
	min_frame_distance: int = 0,
	stable: bool = True,
) -> List[FrameScore]:
	"""Select top-k from an iterable of (frame_index, score) pairs."""

	pairs = list(frame_score_pairs)
	if not pairs or k <= 0:
		return []

	# Sort by score rank. Stable tie-breaker by index.
	if stable:
		pairs.sort(key=lambda p: (float(p[1]), -int(p[0])) if largest else (-float(p[1]), -int(p[0])), reverse=True)
	else:
		pairs.sort(key=lambda p: float(p[1]) if largest else -float(p[1]), reverse=True)

	if min_frame_distance < 0:
		raise ValueError("min_frame_distance must be >= 0")

	selected: List[FrameScore] = []
	taken: List[int] = []

	for frame_idx, score in pairs:
		frame_idx_i = int(frame_idx)
		if min_frame_distance:
			if any(abs(frame_idx_i - t) < min_frame_distance for t in taken):
				continue
		selected.append(FrameScore(frame_index=frame_idx_i, score=float(score)))
		taken.append(frame_idx_i)
		if len(selected) >= k:
			break

	return selected


def as_sorted_by_index(frames: Iterable[FrameScore]) -> List[FrameScore]:
	"""Return frames sorted by frame_index ascending."""

	return sorted(list(frames), key=lambda fs: fs.frame_index)


def as_indices(frames: Iterable[FrameScore]) -> List[int]:
	"""Extract frame indices from FrameScore items."""

	return [fs.frame_index for fs in frames]


def as_scores(frames: Iterable[FrameScore]) -> List[float]:
	"""Extract scores from FrameScore items."""

	return [fs.score for fs in frames]
