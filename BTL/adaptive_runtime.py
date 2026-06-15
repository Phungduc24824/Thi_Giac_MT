"""Adaptive runtime: detect CPU capability and choose model profile.

Usage:
    python adaptive_runtime.py
"""

import logging
from typing import Dict

try:
    import psutil
except Exception:
    psutil = None

try:
    import cpuinfo
except Exception:
    cpuinfo = None


logging.basicConfig(level=logging.INFO)


def get_cpu_info() -> Dict:
    info = {}
    try:
        if psutil:
            info['logical_cores'] = psutil.cpu_count(logical=True)
            info['physical_cores'] = psutil.cpu_count(logical=False)
            try:
                freq = psutil.cpu_freq()
                info['max_freq_mhz'] = freq.max if freq else None
            except Exception:
                info['max_freq_mhz'] = None
            info['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        else:
            info['logical_cores'] = None
            info['physical_cores'] = None
            info['max_freq_mhz'] = None
            info['cpu_percent'] = None

        if cpuinfo:
            ci = cpuinfo.get_cpu_info()
            info['arch'] = ci.get('arch')
            info['bits'] = ci.get('bits')
            info['flags'] = ci.get('flags')
        else:
            info['arch'] = None
            info['bits'] = None
            info['flags'] = None
    except Exception as e:
        logging.exception('Failed to collect CPU info: %s', e)
    return info


def estimate_capacity(info: Dict) -> float:
    """Return a heuristic score for CPU capacity.
    Higher means more capable.
    """
    score = 0.0
    cores = info.get('physical_cores') or info.get('logical_cores') or 0
    score += float(cores) * 1.0
    freq = info.get('max_freq_mhz')
    if freq:
        score += float(freq) / 1000.0
    flags = info.get('flags') or []
    if isinstance(flags, (list, tuple)) and any(f for f in flags if 'avx' in f.lower()):
        score += 1.5
    return score


def choose_model_profile(info: Dict, light_threshold: float = 2.5, heavy_threshold: float = 6.0) -> str:
    """Decide between 'light' and 'heavy' profiles. """
    score = estimate_capacity(info)
    logging.info('CPU capacity score: %.2f', score)
    if score >= heavy_threshold:
        return 'heavy'
    if score <= light_threshold:
        return 'light'
    return 'balanced'


class AdaptiveRuntime:
    def __init__(self, models: Dict[str, str]):
        """models: mapping profile->model_path (e.g. {'light': '...pt', 'heavy': '...pt'})"""
        self.models = models
        self.loaded = None

    def select_profile(self) -> str:
        info = get_cpu_info()
        profile = choose_model_profile(info)
        logging.info('Selected profile: %s', profile)
        return profile

    def load_model(self, profile: str):
        path = self.models.get(profile)
        if not path:
            raise ValueError(f'No model for profile {profile}')
        try:
            import torch
            try:
                model = torch.jit.load(path)
                self.loaded = model
                logging.info('Loaded TorchScript model from %s', path)
                return model
            except Exception:
                # Avoid calling torch.load on untrusted files (security risk).
                logging.warning('Failed to load TorchScript via torch.jit.load; not attempting torch.load for safety. Returning path.')
                self.loaded = path
                return path
        except Exception:
            logging.warning('Torch not available. Returning path only.')
            self.loaded = path
            return path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--light', default='models/light.pt', help='Light model path')
    parser.add_argument('--heavy', default='models/heavy.pt', help='Heavy model path')
    args = parser.parse_args()

    info = get_cpu_info()
    print('CPU info:', info)
    profile = choose_model_profile(info)
    print('Chosen profile:', profile)

    rt = AdaptiveRuntime({'light': args.light, 'heavy': args.heavy})
    chosen = rt.select_profile()
    print('Runtime selected:', chosen)
