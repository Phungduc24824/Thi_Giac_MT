"""Adaptive Inference Manager integrating CPU capability and realtime quality.

Provides a small orchestration layer that selects a model profile per-input
and can call user-provided inference functions for light/heavy models.
"""
from typing import Any, Callable, Dict, Optional
import logging

from adaptive_runtime import AdaptiveRuntime
from signal_quality import (
    assess_image_quality,
    assess_signal_quality,
    RealtimeQualityTracker,
)

logging.basicConfig(level=logging.INFO)


class AdaptiveInferenceManager:
    def __init__(self, models: Dict[str, str], low_quality_threshold: float = 0.45):
        self.runtime = AdaptiveRuntime(models)
        self.tracker = RealtimeQualityTracker(alpha=0.25)
        self.low_quality_threshold = low_quality_threshold

    def decide_profile(self, input_data: Any, prefer_heavy_when_possible: bool = True) -> str:
        # runtime static capability
        static_profile = self.runtime.select_profile()
        # assess dynamic quality
        if hasattr(input_data, 'ndim') and getattr(input_data, 'ndim', 0) >= 1:
            if input_data.ndim == 1:
                score, conf = assess_signal_quality(input_data)
            else:
                score, conf = assess_image_quality(input_data)
        else:
            score, conf = 0.0, 0.2
        score_ewma, conf_ewma = self.tracker.update(score, conf)
        logging.info('Quality score %.3f (conf %.3f), ewma %.3f (conf_ewma %.3f)', score, conf, score_ewma, conf_ewma)

        if score_ewma < self.low_quality_threshold:
            return 'light'
        if static_profile == 'heavy' and score_ewma >= self.low_quality_threshold and prefer_heavy_when_possible:
            return 'heavy'
        return 'balanced'

    def run(self, input_data: Any, run_light: Callable, run_heavy: Callable, run_balanced: Optional[Callable] = None) -> Any:
        profile = self.decide_profile(input_data)
        logging.info('Running profile: %s', profile)
        if profile == 'light':
            return run_light(input_data)
        if profile == 'heavy':
            return run_heavy(input_data)
        if run_balanced is not None:
            return run_balanced(input_data)
        # fallback to light
        return run_light(input_data)


if __name__ == '__main__':
    # Example usage with dummy runner functions
    import numpy as np

    def light_fn(x):
        return {'profile': 'light', 'len': getattr(x, 'size', None)}

    def heavy_fn(x):
        return {'profile': 'heavy', 'len': getattr(x, 'size', None)}

    manager = AdaptiveInferenceManager({'light': 'models/light.pt', 'heavy': 'models/heavy.pt'})
    # Use a fixed seed for the example to make behavior deterministic
    rng = np.random.default_rng(42)
    dummy = rng.standard_normal(256)
    out = manager.run(dummy, light_fn, heavy_fn)
    print(out)
