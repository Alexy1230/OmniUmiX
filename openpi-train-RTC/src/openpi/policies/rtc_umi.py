"""Stateless RTC transport adapter for existing JAX UMI checkpoints."""

import time
from types import MethodType
import numpy as np
import jax
import jax.numpy as jnp
from openpi.models.rtc_sampler import sample_actions_rtc


def normalize_prefix(absolute, reference, stats, action_dim):
    """Rebase physical targets, then use checkpoint quantiles; gripper stays absolute."""
    from openpi import transforms_umi as geom

    absolute = np.asarray(absolute, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    width = reference.size
    if reference.shape != (width,) or width not in (10, 20):
        raise ValueError("RTC reference must be [10] or [20]")
    if absolute.ndim != 2 or absolute.shape[1] != width or action_dim < width:
        raise ValueError("RTC absolute prefix has wrong shape")
    if not np.isfinite(absolute).all() or not np.isfinite(reference).all():
        raise ValueError("Non-finite RTC prefix/reference")
    result = absolute.copy()
    for start in range(0, width, 10):
        target = geom.pose9_to_matrix(absolute[:, start : start + 9])
        base = geom.pose9_to_matrix(reference[start : start + 9])
        result[:, start : start + 9] = geom.matrix_to_pose9(
            np.linalg.inv(base) @ target
        )
    q01, q99 = np.asarray(stats.q01), np.asarray(stats.q99)
    if (
        q01.shape != (width,)
        or q99.shape != (width,)
        or not np.isfinite(q01).all()
        or not np.isfinite(q99).all()
        or np.any(q99 < q01)
    ):
        raise ValueError(
            "RTC requires finite checkpoint action quantiles matching arm count"
        )
    for start in range(0, width, 10):
        for dim in (start, start + 1, start + 2, start + 9):
            result[:, dim] = (result[:, dim] - q01[dim]) / (
                q99[dim] - q01[dim] + 1e-6
            ) * 2 - 1
    return np.pad(result, ((0, 0), (0, action_dim - width)))


class RTCUmiPolicy:
    def __init__(
        self,
        base,
        stats,
        num_robots,
        *,
        normalization_mode,
        num_steps=10,
        max_guidance=5.0,
        training_fps=30.0,
        execution_fps=30.0,
    ):
        from openpi.models.pi0 import Pi0
        from openpi.shared.nnx_utils import module_jit

        if (
            base._is_pytorch_model
            or not isinstance(base._model, Pi0)
            or not base._model.pi05
        ):
            raise ValueError("RTC currently supports JAX Pi05 UMI checkpoints only")
        supported_modes = {"umi_action10": 1, "umi_action20": 2, "umi_proprio": 1}
        if normalization_mode not in supported_modes:
            raise ValueError(f"Unsupported RTC normalization mode: {normalization_mode}")
        if num_robots != supported_modes[normalization_mode]:
            raise ValueError("RTC arm count does not match normalization mode")
        if normalization_mode == "umi_proprio" and (
            base.metadata.get("proprioception") is not True
            or base.metadata.get("state_dim") != 32
            or base.metadata.get("image_history_horizon") != 2
        ):
            raise ValueError("Proprio checkpoint metadata is incomplete or incompatible")
        if (
            num_robots not in (1, 2)
            or num_steps <= 0
            or not 0 < max_guidance <= 20
            or not np.isfinite(training_fps)
            or training_fps <= 0
            or not np.isfinite(execution_fps)
            or execution_fps <= 0
        ):
            raise ValueError("Invalid RTC server settings")
        self.base, self.stats, self.num_robots = base, stats, num_robots
        self.normalization_mode = normalization_mode
        self.num_steps, self.max_guidance = num_steps, max_guidance
        self.horizon, self.dim = base._model.action_horizon, base._model.action_dim
        self.sample = module_jit(MethodType(sample_actions_rtc, base._model))
        self._metadata = dict(
            base.metadata,
            rtc_version=1,
            num_robots=num_robots,
            action_dim=10 * num_robots,
            action_horizon=self.horizon,
            is_relative=False,
            training_fps=float(training_fps),
            execution_fps=float(execution_fps),
            time_scale=float(training_fps / execution_fps),
            # Compatibility field consumed by existing clients: this is the
            # rate used to timestamp and execute returned action rows.
            fps=float(execution_fps),
            image_keys=(
                ["observation/image"]
                if num_robots == 1
                else ["observation/image_right", "observation/image_left"]
            ),
        )

    @property
    def metadata(self):
        return self._metadata

    def infer(self, obs):
        from openpi.models import model as model_api

        request = dict(obs)
        rtc = request.pop("rtc", None)
        if rtc is None:
            return self.base.infer(request)
        if rtc.get("version") != 1:
            raise ValueError("Unsupported RTC protocol")
        inputs = self.base._input_transform(request)
        reference = inputs["current_tcp_pose9_gripper"]
        if np.asarray(reference).shape != (10 * self.num_robots,):
            raise ValueError("RTC arm count does not match checkpoint")
        absolute = np.asarray(rtc["prefix_absolute"], dtype=np.float32)
        weights = np.asarray(rtc["weights"], dtype=np.float32)
        if absolute.shape != (self.horizon, 10 * self.num_robots):
            raise ValueError("RTC prefix must be full-horizon absolute pose9/gripper")
        if (
            weights.shape != (self.horizon,)
            or not np.isfinite(weights).all()
            or np.any((weights < 0) | (weights > 1))
        ):
            raise ValueError("RTC weights must be finite [H] in [0,1]")
        target = normalize_prefix(absolute, reference, self.stats["actions"], self.dim)
        mask = np.zeros_like(target)
        mask[:, : 10 * self.num_robots] = weights[
            :, None
        ]  # Never guide padding dimensions.
        batch = jax.tree.map(lambda v: jnp.asarray(v)[None], inputs)
        self.base._rng, rng = jax.random.split(self.base._rng)
        started = time.monotonic()
        prediction = self.sample(
            rng,
            model_api.Observation.from_dict(batch),
            prefix=jnp.asarray(target)[None],
            weights=jnp.asarray(mask)[None],
            num_steps=self.num_steps,
            max_guidance=self.max_guidance,
        )
        prediction = np.asarray(
            prediction[0]
        )  # Synchronize before reporting inference time.
        if not np.isfinite(prediction).all():
            raise ValueError("RTC sampler produced non-finite actions")
        output = self.base._output_transform(
            dict(
                actions=prediction,
                state=inputs["state"],
                current_tcp_pose9_gripper=reference,
            )
        )
        output["rtc_version"] = 1
        output["policy_timing"] = {"infer_ms": (time.monotonic() - started) * 1000}
        return output
