"""Compatibility helpers for old saved Stable-Baselines3 models."""

import sys


def apply_compat_shims() -> None:
    """Patch common NumPy/SB3 pickle path changes.

    This does not change training. It only makes old model loading less fragile
    across NumPy/SB3 versions.
    """
    try:
        import numpy.core.numeric as numeric
        import numpy.core.multiarray as multiarray
        import numpy.core.umath as umath
        import numpy.core as core
        sys.modules.setdefault("numpy._core", core)
        sys.modules.setdefault("numpy._core.numeric", numeric)
        sys.modules.setdefault("numpy._core.multiarray", multiarray)
        sys.modules.setdefault("numpy._core.umath", umath)
    except Exception:
        pass

    try:
        import stable_baselines3.common.utils as sb3_utils

        for name in ("FloatSchedule", "ConstantSchedule", "LinearSchedule"):
            if not hasattr(sb3_utils, name):
                class DummySchedule:
                    def __init__(self, value=0.0, *args, **kwargs):
                        self.value = float(value)

                    def __call__(self, progress_remaining=1.0):
                        return float(getattr(self, "value", 0.0))

                    def __setstate__(self, state):
                        if isinstance(state, dict):
                            self.__dict__.update(state)
                        else:
                            self.value = float(state)

                setattr(sb3_utils, name, DummySchedule)
    except Exception:
        pass

    try:
        from gymnasium.spaces.space import Space

        def patched_setstate(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

        Space.__setstate__ = patched_setstate
    except Exception:
        pass
