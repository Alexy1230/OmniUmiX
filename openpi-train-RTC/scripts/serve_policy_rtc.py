"""Serve an existing UMI checkpoint with optional inference-only RTC guidance."""

import argparse
from pathlib import Path
import math
import numpy as np
from openpi.policies.policy_config import create_trained_policy
from openpi.policies.rtc_umi import RTCUmiPolicy
from openpi.training.config import get_config
from openpi.training.checkpoints import load_norm_stats
from openpi.shared.download import maybe_download
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


def resolve_timing(metadata, training_fps_check=None, execution_fps=None):
    training_fps = metadata.get("fps") if isinstance(metadata, dict) else None
    if (
        isinstance(training_fps, bool)
        or not isinstance(training_fps, (int, float))
        or not math.isfinite(training_fps)
        or training_fps <= 0
    ):
        raise ValueError("Checkpoint policy metadata must contain a positive finite fps")
    if training_fps_check is not None and (
        not math.isfinite(training_fps_check)
        or training_fps_check <= 0
        or not np.isclose(training_fps_check, training_fps)
    ):
        raise ValueError(
            f"--fps={training_fps_check} does not match checkpoint metadata fps={training_fps}"
        )
    execution_fps = training_fps if execution_fps is None else execution_fps
    if (
        isinstance(execution_fps, bool)
        or not isinstance(execution_fps, (int, float))
        or not math.isfinite(execution_fps)
        or execution_fps <= 0
    ):
        raise ValueError("--execution-fps must be positive and finite")
    return float(training_fps), float(execution_fps), float(training_fps / execution_fps)


def resolve_fps(metadata, requested):
    """Backward-compatible checkpoint frequency validation."""
    return resolve_timing(metadata, requested)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--port", type=int, default=8099)
    p.add_argument(
        "--fps",
        type=float,
        help="Optional training-fps check; checkpoint metadata is authoritative",
    )
    p.add_argument(
        "--execution-fps",
        type=float,
        help="Runtime action rate; defaults to checkpoint training fps",
    )
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--max-guidance", type=float, default=5.0)
    p.add_argument("--default-prompt")
    args = p.parse_args()
    config = get_config(args.config)
    data = config.data.create(config.assets_dirs, config.model)
    if data.normalization_mode not in ("umi_action10", "umi_action20", "umi_proprio"):
        p.error("Use a UMI single-arm, bimanual, or single-arm proprio config")
    checkpoint = Path(maybe_download(args.checkpoint))
    stats = load_norm_stats(checkpoint / "assets", data.asset_id)
    base = create_trained_policy(
        config, checkpoint, norm_stats=stats, default_prompt=args.default_prompt
    )
    try:
        training_fps, execution_fps, time_scale = resolve_timing(
            base.metadata, args.fps, args.execution_fps
        )
    except ValueError as exc:
        p.error(str(exc))
    print(
        f"RTC timing: training={training_fps:g} Hz, execution={execution_fps:g} Hz, "
        f"time_scale={time_scale:g}x"
    )
    policy = RTCUmiPolicy(
        base,
        stats,
        2 if data.normalization_mode == "umi_action20" else 1,
        normalization_mode=data.normalization_mode,
        num_steps=args.num_steps,
        max_guidance=args.max_guidance,
        training_fps=training_fps,
        execution_fps=execution_fps,
    )
    WebsocketPolicyServer(
        policy, port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
