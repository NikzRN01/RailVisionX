from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.quality.blur_score import mean_gradient_magnitude, variance_of_laplacian
from src.quality.lowlight_score import mean_intensity


@dataclass(frozen=True)
class RestorationMetrics:
	lap_var_in: float
	lap_var_out: float
	lap_var_delta: float
	grad_mean_in: float
	grad_mean_out: float
	grad_mean_delta: float
	intensity_in: float
	intensity_out: float
	intensity_delta: float
	correction_score: float

	def as_dict(self) -> dict[str, Any]:
		return {
			"lap_var_in": self.lap_var_in,
			"lap_var_out": self.lap_var_out,
			"lap_var_delta": self.lap_var_delta,
			"grad_mean_in": self.grad_mean_in,
			"grad_mean_out": self.grad_mean_out,
			"grad_mean_delta": self.grad_mean_delta,
			"intensity_in": self.intensity_in,
			"intensity_out": self.intensity_out,
			"intensity_delta": self.intensity_delta,
			"correction_score": self.correction_score,
		}


def restoration_metrics(blurred: np.ndarray, restored: np.ndarray) -> RestorationMetrics:
	"""Score restoration quality from input/output only.

	This does NOT require ground truth.
	The main idea is to measure sharpness increase (edges) from blurred -> restored.
	"""
	lap_in = variance_of_laplacian(blurred)
	lap_out = variance_of_laplacian(restored)
	grad_in = mean_gradient_magnitude(blurred)
	grad_out = mean_gradient_magnitude(restored)
	int_in = mean_intensity(blurred)
	int_out = mean_intensity(restored)

	lap_delta = lap_out - lap_in
	grad_delta = grad_out - grad_in
	int_delta = int_out - int_in

	# Simple combined score; weights keep the magnitudes reasonable.
	correction = float(lap_delta + 10.0 * grad_delta)
	return RestorationMetrics(
		lap_var_in=lap_in,
		lap_var_out=lap_out,
		lap_var_delta=lap_delta,
		grad_mean_in=grad_in,
		grad_mean_out=grad_out,
		grad_mean_delta=grad_delta,
		intensity_in=int_in,
		intensity_out=int_out,
		intensity_delta=int_delta,
		correction_score=correction,
	)
