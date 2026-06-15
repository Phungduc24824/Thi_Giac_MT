"""Simple tests for the adaptive manager and signal/image quality hooks."""
import numpy as np

from adaptive_manager import AdaptiveInferenceManager


def test_dummy_runners():
    models = {'light': 'models/light.pt', 'heavy': 'models/heavy.pt'}
    manager = AdaptiveInferenceManager(models)

    def light_fn(x):
        return {'profile': 'light', 'score': getattr(x, 'shape', None)}

    def heavy_fn(x):
        return {'profile': 'heavy', 'score': getattr(x, 'shape', None)}

    # Simulate low-quality image (blurry)
    low_quality_img = np.zeros((64, 64), dtype=np.uint8)
    out1 = manager.run(low_quality_img, light_fn, heavy_fn)
    print('Low-quality img ->', out1)

    # Simulate high-quality image (random texture) using Generator API
    # Fixed seed to make test deterministic
    rng = np.random.default_rng(42)
    high_quality_img = (rng.standard_normal((224, 224, 3)) * 64 + 128).clip(0, 255).astype(np.uint8)
    out2 = manager.run(high_quality_img, light_fn, heavy_fn)
    print('High-quality img ->', out2)

    # Simulate good 1D signal
    sig = np.sin(np.linspace(0, 10 * np.pi, 1024))
    out3 = manager.run(sig, light_fn, heavy_fn)
    print('Signal ->', out3)

    # Basic assertions (profiles should be strings)
    assert isinstance(out1['profile'], str)
    assert isinstance(out2['profile'], str)
    assert isinstance(out3['profile'], str)


if __name__ == '__main__':
    test_dummy_runners()
    print('Tests completed.')
