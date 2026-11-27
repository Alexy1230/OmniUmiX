"""Offline UMI bimanual evaluation using the configured OpenPI training dataset.

Run with --help for options. Only writes new files under --output-dir. This is
teacher-anchored, open-loop training-set evaluation, not robot success evaluation.
"""

import argparse
import copy
import csv
import dataclasses
import datetime
import json
import logging
import os
from pathlib import Path

import numpy as np


def chunk_errors(prediction, target):
    """Per-timestep physical errors, keeping the two robot base frames separate."""
    from openpi.transforms_umi import pose9_to_matrix

    errors = {}
    for arm, start in (("right", 0), ("left", 10)):
        pred = prediction[:, start : start + 10]
        truth = target[:, start : start + 10]
        errors[f"{arm}_position_mm"] = np.linalg.norm(pred[:, :3] - truth[:, :3], axis=-1) * 1000
        pred_rotation = pose9_to_matrix(pred[:, :9])[:, :3, :3]
        true_rotation = pose9_to_matrix(truth[:, :9])[:, :3, :3]
        cosine = (np.einsum("tij,tij->t", pred_rotation, true_rotation) - 1) / 2
        errors[f"{arm}_rotation_deg"] = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
        errors[f"{arm}_gripper_abs"] = np.abs(pred[:, 9] - truth[:, 9])
    return errors


def summarize_errors(error_chunks, masks):
    result = {}
    for key in error_chunks[0]:
        values = np.concatenate([errors[key][mask] for errors, mask in zip(error_chunks, masks, strict=True)])
        result[key] = {
            "mean": float(values.mean()),
            "rmse": float(np.sqrt(np.mean(values**2))),
            "p95": float(np.percentile(values, 95)),
        }
    return result


def plot_chunk(path, target, prediction, valid, errors, fps, title):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig = plt.figure(figsize=(15, 17), layout="constrained")
    grid = fig.add_gridspec(7, 2)
    times = np.arange(len(valid))[valid] / fps
    for col, (arm, start) in enumerate((("right", 0), ("left", 10))):
        gt, pred = target[valid, start : start + 10], prediction[valid, start : start + 10]
        ax = fig.add_subplot(grid[:2, col], projection="3d")
        ax.plot(*gt[:, :3].T, label="ground truth", color="tab:blue")
        ax.plot(*pred[:, :3].T, label="prediction", color="tab:orange")
        ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title=f"{arm}: own robot base")
        ax.legend()
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(4))
        for row, (dim, label) in enumerate(
            ((0, "x (m)"), (1, "y (m)"), (2, "z (m)"), (9, "gripper (dataset units)")), 2
        ):
            ax = fig.add_subplot(grid[row, col])
            ax.plot(times, gt[:, dim], color="tab:blue")
            ax.plot(times, pred[:, dim], color="tab:orange")
            ax.set(xlabel="prediction horizon (s)", ylabel=label)
            ax.grid(alpha=0.3)
        ax = fig.add_subplot(grid[6, col])
        ax.plot(times, errors[f"{arm}_rotation_deg"][valid])
        ax.set(xlabel="prediction horizon (s)", ylabel="rotation error (deg)")
        ax.grid(alpha=0.3)
    fig.suptitle(title + "\nGT first-action anchor; open-loop training-set evaluation")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def start_recording(path):
    import rerun as rr
    import rerun.blueprint as rrb

    rr.init("umi_bimanual_offline_evaluation", recording_id=path.stem, spawn=False)
    rr.save(path)
    rr.send_blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="right", name="Right robot base"),
                rrb.Spatial3DView(origin="left", name="Left robot base"),
            ),
            rrb.Horizontal(
                rrb.Spatial2DView(origin="images/right", name="Right wrist"),
                rrb.Spatial2DView(origin="images/left", name="Left wrist"),
            ),
            rrb.TimeSeriesView(origin="errors", name="Mean chunk errors"),
        )
    )
    rr.log(
        "description",
        rr.TextDocument(
            "Blue: ground truth. Orange: prediction. Each arm is in its own base frame. "
            "Scrub frame_index to select an observation and its future action chunk. "
            "GT first-action anchors are used; this is not a closed-loop rollout."
        ),
        static=True,
    )
    return rr


def plot_overview(path, rows, error_chunks, masks):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for episode in dict.fromkeys(row["episode"] for row in rows):
        selected = [row for row in rows if row["episode"] == episode]
        frames = [row["frame"] for row in selected]
        if selected[0]["flow_loss"] is not None:
            axes[0, 0].plot(frames, [row["flow_loss"] for row in selected], ".-", label=f"episode {episode}")
        for arm in ("right", "left"):
            axes[0, 1].plot(frames, [row[f"{arm}_position_mm"] for row in selected], ".-", label=f"ep {episode} {arm}")
    axes[0, 0].set(
        title="Evaluation flow loss (not historical training loss)", xlabel="observation frame", ylabel="MSE"
    )
    axes[0, 1].set(title="Mean chunk position error", xlabel="observation frame", ylabel="mm")
    for arm in ("right", "left"):
        for ax, suffix, unit in ((axes[1, 0], "position_mm", "mm"), (axes[1, 1], "rotation_deg", "degrees")):
            values = []
            for step in range(len(masks[0])):
                selected = [
                    error[f"{arm}_{suffix}"][step]
                    for error, mask in zip(error_chunks, masks, strict=True)
                    if mask[step]
                ]
                values.append(np.mean(selected) if selected else np.nan)
            ax.plot(values, label=arm)
            ax.set(xlabel="prediction horizon step", ylabel=unit, title="Mean error by prediction horizon")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        if ax.lines:
            ax.legend()
    fig.suptitle("Open-loop training-set evaluation; padding excluded")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def log_chunk(rr, frame, timestamp, inputs, target, prediction, valid, errors):
    from openpi.policies.umi_policy import _parse_image

    rr.set_time("frame_index", sequence=frame)
    rr.set_time("timestamp", duration=timestamp)
    for arm, start in (("right", 0), ("left", 10)):
        rr.log(f"images/{arm}", rr.Image(_parse_image(inputs[f"observation/image_{arm}"])))
        for label, actions, color in (
            ("ground_truth", target, [60, 130, 255]),
            ("prediction", prediction, [255, 140, 30]),
        ):
            points = actions[valid, start : start + 3]
            rr.log(f"{arm}/{label}/trajectory", rr.LineStrips3D([points], colors=[color]))
            rr.log(f"{arm}/{label}/points", rr.Points3D(points, colors=[color], radii=0.002))
        for suffix in ("position_mm", "rotation_deg", "gripper_abs"):
            key = f"{arm}_{suffix}"
            rr.log(f"errors/{key}", rr.Scalars(float(errors[key][valid].mean())))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default="infra_test_pi05_biumi_pick_place_cup_rod")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Specific checkpoint step directory (e.g. .../29999)"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Must not already exist; defaults to outputs/umi_eval/<timestamp>"
    )
    episode_group = parser.add_mutually_exclusive_group()
    episode_group.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2], help="Episode IDs to evaluate")
    episode_group.add_argument(
        "--all-episodes", action="store_true", help="Evaluate every episode in the configured dataset"
    )
    parser.add_argument("--stride", type=int, default=32, help="Frames between evaluated observations")
    parser.add_argument("--max-chunks", type=int, default=0, help="Maximum chunks per episode; 0 means all")
    parser.add_argument("--plots-per-episode", type=int, default=3, help="Number of selected chunks to plot as PNG")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--loss-repeats", type=int, default=1, help="Monte Carlo repeats of compute_loss(train=False)")
    parser.add_argument("--skip-loss", action="store_true", help="Only evaluate sampled trajectories")
    parser.add_argument("--no-rerun", action="store_true", help="Skip comparison .rrd export")
    parser.add_argument(
        "--export-dataset", action="store_true", help="Also invoke LeRobot's built-in raw episode viewer/exporter"
    )
    args = parser.parse_args()
    if min(args.stride, args.loss_repeats, args.denoise_steps) < 1 or min(args.max_chunks, args.plots_per_episode) < 0:
        parser.error("stride/loss-repeats/denoise-steps must be positive; max-chunks/plots-per-episode nonnegative")
    if len(set(args.episodes)) != len(args.episodes):
        parser.error("episode IDs must be unique")
    return args


def main():
    args = parse_args()
    # Avoid reserving nearly all free GPU memory on a shared training server.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    from openpi import transforms
    from openpi.models import model as model_lib
    from openpi.policies import policy_config
    from openpi.shared import nnx_utils
    from openpi.training import checkpoints
    from openpi.training import config
    from openpi.training import data_loader

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    train = config.get_config(args.config)
    data = train.data.create(train.assets_dirs, train.model)
    if data.normalization_mode != "umi_action20":
        raise ValueError("This evaluator requires the UMI bimanual umi_action20 configuration.")
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "params").is_dir():
        raise ValueError("Expected a JAX OpenPI checkpoint with a params directory; pass a specific saved step.")
    meta = LeRobotDatasetMetadata(data.repo_id)
    if args.all_episodes:
        args.episodes = sorted(meta.episodes)
    missing = set(args.episodes) - set(meta.episodes)
    if missing:
        raise ValueError(f"Unknown episodes: {sorted(missing)}")
    norm_stats = checkpoints.load_norm_stats(checkpoint / "assets", data.asset_id)
    data = dataclasses.replace(data, norm_stats=norm_stats)
    dataset = data_loader.create_torch_dataset(data, train.model.action_horizon, train.model)
    repack = transforms.compose(data.repack_transforms.inputs)
    starts, cursor = {}, 0
    for episode, info in sorted(meta.episodes.items()):
        starts[episode] = cursor
        cursor += info["length"]
    if cursor != len(dataset):
        raise ValueError("Episode metadata and dataset length disagree.")
    output = args.output_dir or Path("outputs/umi_eval") / datetime.datetime.now(datetime.UTC).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "evaluation": "training_set_open_loop_gt_first_action_anchor",
        "config": args.config,
        "default_prompt": getattr(train.data, "default_prompt", None),
        "dataset": data.repo_id,
        "checkpoint": str(checkpoint),
        "fps": meta.fps,
        "episodes": args.episodes,
        "stride": args.stride,
        "seed": args.seed,
        "denoise_steps": args.denoise_steps,
        "max_chunks_per_episode": args.max_chunks,
        "loss_repeats": 0 if args.skip_loss else args.loss_repeats,
        "action_order": "right xyz/column-rot6d/gripper, left xyz/column-rot6d/gripper",
        "position_units": "meters in saved arrays; millimeters in error metrics",
        "gripper_units": "unchanged dataset units",
        "reference": "GT action chunk first frame; never supplied as measured model state",
        "loss_definition": "flow-matching MSE over all internal model action dimensions (including padding), train=False",
        "weighting": "valid chunk timesteps; overlapping windows count repeatedly when stride < horizon",
        "historical_training_loss": "not reconstructed by this evaluation",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logging.info("Loading checkpoint %s; results -> %s", checkpoint, output)
    policy = policy_config.create_trained_policy(
        train, checkpoint, norm_stats=norm_stats, sample_kwargs={"num_steps": args.denoise_steps}
    )
    # No public loss API on Policy: reuse its already-loaded model without a second checkpoint allocation.
    loss_fn = None if args.skip_loss else nnx_utils.module_jit(policy._model.compute_loss, static_argnames=("train",))  # noqa: SLF001
    rows, all_errors, all_masks, all_losses = [], [], [], []
    for episode in args.episodes:
        length = meta.episodes[episode]["length"]
        frames = list(range(0, length, args.stride))
        if args.max_chunks:
            frames = frames[: args.max_chunks]
        plot_frames = set(
            np.array(frames)[np.linspace(0, len(frames) - 1, min(len(frames), args.plots_per_episode), dtype=int)]
        )
        rr = None if args.no_rerun else start_recording(output / f"episode_{episode:06d}.rrd")
        records = {
            key: [] for key in ("prediction", "target", "valid", "reference", "frame_index", "timestamp", "flow_loss")
        }
        for frame in frames:
            index = starts[episode] + frame
            raw = dataset[index]
            if int(raw["episode_index"]) != episode or int(raw["frame_index"]) != frame:
                raise ValueError("Dataset row does not match requested episode/frame.")
            inputs = repack(copy.deepcopy(raw))
            target = np.asarray(inputs.pop("actions"), dtype=np.float32)
            expected = (train.model.action_horizon, 20)
            if target.shape != expected:
                raise ValueError(f"Expected target {expected}, got {target.shape}")
            pad_key = data.action_sequence_keys[0] + "_is_pad"
            valid = ~np.asarray(raw[pad_key], dtype=bool)
            if valid.shape != (expected[0],) or not valid[0]:
                raise ValueError("Invalid LeRobot action padding mask")
            reference = target[0].copy()
            inputs["current_tcp_pose9_gripper"] = reference
            # Per-observation seeds make results reproducible across episode subsets/checkpoints.
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, episode, frame]))
            noise = rng.standard_normal((train.model.action_horizon, train.model.action_dim)).astype(np.float32)
            prediction = np.asarray(policy.infer(inputs, noise=noise)["actions"])
            if prediction.shape != expected or not np.isfinite(prediction).all():
                raise ValueError("Invalid model prediction")
            errors = chunk_errors(prediction, target)
            flow_loss = np.full((train.model.action_horizon,), np.nan)
            if loss_fn is not None:
                sample = data_loader.transform_dataset([copy.deepcopy(raw)], data)[0]
                action = jnp.asarray(sample.pop("actions"))[None]
                obs = model_lib.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None], sample))
                base_key = jax.random.fold_in(jax.random.key(args.seed), index)
                losses = [
                    np.asarray(loss_fn(jax.random.fold_in(base_key, repeat), obs, action, train=False))[0]
                    for repeat in range(args.loss_repeats)
                ]
                flow_loss = np.mean(losses, axis=0)
                if not np.isfinite(flow_loss).all():
                    raise ValueError("Non-finite evaluation loss")
                all_losses.append(flow_loss[valid])
            timestamp = float(raw["timestamp"])
            for key, value in (
                ("prediction", prediction),
                ("target", target),
                ("valid", valid),
                ("reference", reference),
                ("frame_index", frame),
                ("timestamp", timestamp),
                ("flow_loss", flow_loss),
            ):
                records[key].append(value)
            row = {
                "episode": episode,
                "frame": frame,
                "valid_steps": int(valid.sum()),
                "flow_loss": None if loss_fn is None else float(flow_loss[valid].mean()),
            }
            row.update({key: float(values[valid].mean()) for key, values in errors.items()})
            rows.append(row)
            all_errors.append(errors)
            all_masks.append(valid)
            if rr is not None:
                log_chunk(rr, frame, timestamp, inputs, target, prediction, valid, errors)
            if frame in plot_frames:
                plot_chunk(
                    output / f"episode_{episode:06d}_frame_{frame:06d}.png",
                    target,
                    prediction,
                    valid,
                    errors,
                    meta.fps,
                    f"Episode {episode}, frame {frame}, checkpoint {checkpoint.name}",
                )
            logging.info(
                "episode=%d frame=%d valid=%d flow_loss=%s right=%.2fmm left=%.2fmm",
                episode,
                frame,
                valid.sum(),
                row["flow_loss"],
                row["right_position_mm"],
                row["left_position_mm"],
            )
        np.savez_compressed(
            output / f"episode_{episode:06d}.npz", **{k: np.stack(v) for k, v in records.items()}, fps=meta.fps
        )
        if rr is not None:
            rr.disconnect()
    with (output / "chunks.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "chunks": len(rows),
        "valid_chunk_timesteps": int(sum(mask.sum() for mask in all_masks)),
        "flow_loss": None if not all_losses else float(np.concatenate(all_losses).mean()),
        "errors": summarize_errors(all_errors, all_masks),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot_overview(output / "overview.png", rows, all_errors, all_masks)
    # Aggregated horizon errors help reveal drift without stitching open-loop chunks into a rollout.
    with (output / "horizon_errors.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["horizon_step", "count", *all_errors[0]])
        writer.writeheader()
        for step in range(train.model.action_horizon):
            selected = [error for error, mask in zip(all_errors, all_masks, strict=True) if mask[step]]
            if selected:
                writer.writerow(
                    {
                        "horizon_step": step,
                        "count": len(selected),
                        **{key: float(np.mean([error[key][step] for error in selected])) for key in selected[0]},
                    }
                )
    if args.export_dataset:
        from lerobot.scripts.visualize_dataset import visualize_dataset

        # Built-in viewer expects one action per frame, not training chunks.
        raw_dataset = LeRobotDataset(data.repo_id, tolerance_s=0.5 / meta.fps)
        for episode in args.episodes:
            visualize_dataset(raw_dataset, episode, num_workers=0, save=True, output_dir=output / "raw_dataset")
    logging.info("Finished. Summary: %s", json.dumps(summary))
    print(f"Results saved to {output.resolve()}")


if __name__ == "__main__":
    main()
