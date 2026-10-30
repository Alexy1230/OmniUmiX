"""Linear calibration between absolute jaw opening (metres) and motor/model radians."""
import math


def width_to_rad_parameters(config):
    """Return slope, intercept and limits, or None for the legacy conversion."""
    if config is None:
        return None
    keys = ('min_abs_width_m', 'max_abs_width_m', 'min_rad', 'max_rad')
    if not isinstance(config, dict) or any(key not in config for key in keys):
        raise ValueError('width_to_rad requires ' + ', '.join(keys))
    try:
        lo, hi, rlo, rhi = (float(config[key]) for key in keys)
    except (TypeError, ValueError) as exc:
        raise ValueError('width_to_rad endpoints must be finite numbers') from exc
    if not all(math.isfinite(v) for v in (lo, hi, rlo, rhi)):
        raise ValueError('width_to_rad endpoints must be finite numbers')
    if not 0 <= lo < hi or not rlo < rhi:
        raise ValueError('width_to_rad requires 0 <= min width < max width and min_rad < max_rad')
    scale = (rhi - rlo) / (hi - lo)
    offset = rlo - scale * lo
    if not math.isfinite(scale) or not math.isfinite(offset) or scale <= 0:
        raise ValueError('width_to_rad produces an invalid scale or offset')
    return scale, offset, lo, hi
