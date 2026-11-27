"""Continuously query a UMI policy server over one WebSocket connection."""

import dataclasses
import math
import pathlib
import time

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import main as _single_client
import numpy as np
from openpi_client import websocket_client_policy
import tyro


@dataclasses.dataclass
class Args:
    """Arguments for continuous sequential inference."""

    # Local LeRobot dataset directory or Hugging Face repo id.
    dataset_path: str

    # Global dataset frame used by the first request.
    frame_index: int = 0

    # Number of sequential requests sent over the same WebSocket connection.
    num_requests: int = 5

    # Number of dataset frames advanced between requests.
    frame_step: int = 1

    # Requests excluded from steady-state latency statistics.
    warmup_requests: int = 1

    # Extra delay after each response before preparing the next request.
    request_interval_s: float = 0.0

    # Language instruction sent to the policy.
    prompt: str = "Place the red cup on the yellow plate."

    # Policy server hostname or IP address.
    host: str = "127.0.0.1"

    # Policy server WebSocket port.
    port: int = 8000

    # Dataset key containing the fisheye image.
    image_key: str = "observation.images.fisheye_img"

    # Dataset key containing current TCP pose9 + gripper.
    tcp_reference_key: str = "observation.state"

    # If set, save the final request's prediction/GT comparison to this Rerun file.
    visualization_path: pathlib.Path | None = None


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values), percentile))


def _print_latency_summary(name: str, values: list[float]) -> None:
    if not values:
        return
    print(
        f"{name}: mean={np.mean(values):.2f} ms, "
        f"p50={_percentile(values, 50):.2f} ms, "
        f"p95={_percentile(values, 95):.2f} ms, "
        f"min={np.min(values):.2f} ms, max={np.max(values):.2f} ms"
    )


def main(args: Args) -> None:
    if args.num_requests <= 0:
        raise ValueError(f"num_requests must be positive, got {args.num_requests}.")
    if args.frame_step <= 0:
        raise ValueError(f"frame_step must be positive, got {args.frame_step}.")
    if not 0 <= args.warmup_requests < args.num_requests:
        raise ValueError("warmup_requests must be non-negative and smaller than num_requests.")
    if args.request_interval_s < 0:
        raise ValueError("request_interval_s must be non-negative.")

    dataset = LeRobotDataset(args.dataset_path)
    first_frame_index = args.frame_index if args.frame_index >= 0 else len(dataset) + args.frame_index
    last_frame_index = first_frame_index + (args.num_requests - 1) * args.frame_step
    if not 0 <= first_frame_index < len(dataset) or last_frame_index >= len(dataset):
        raise IndexError(
            f"Requested frames {first_frame_index}..{last_frame_index}, "
            f"but the dataset contains {len(dataset)} frames."
        )

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = client.get_server_metadata()
    fps = float(metadata.get("fps", 30))
    action_horizon = int(metadata.get("action_horizon", 32))
    horizon_ms = action_horizon / fps * 1000

    print(f"Connected to ws://{args.host}:{args.port}")
    print(f"Server metadata: {metadata}")
    print(f"Action horizon duration: {action_horizon}/{fps:g} = {horizon_ms:.2f} ms")
    print(
        f"Sending {args.num_requests} sequential requests; "
        f"the first {args.warmup_requests} request(s) are warm-up."
    )

    round_trip_times: list[float] = []
    server_times: list[float] = []
    policy_times: list[float] = []
    final_actions: np.ndarray | None = None
    final_ground_truth: np.ndarray | None = None
    final_tcp_reference: np.ndarray | None = None

    for request_index in range(args.num_requests):
        frame_index = first_frame_index + request_index * args.frame_step
        frame = dataset[frame_index]
        missing_keys = [key for key in (args.image_key, args.tcp_reference_key) if key not in frame]
        if missing_keys:
            raise KeyError(f"Dataset frame is missing {missing_keys}. Available keys: {sorted(frame)}")

        image = _single_client._prepare_image(frame[args.image_key])  # noqa: SLF001
        tcp_reference = _single_client._prepare_tcp_reference(  # noqa: SLF001
            frame[args.tcp_reference_key]
        )
        request = {
            "observation/image": image,
            "current_tcp_pose9_gripper": tcp_reference,
            "prompt": args.prompt,
        }

        start = time.perf_counter()
        response = client.infer(request)
        round_trip_ms = (time.perf_counter() - start) * 1000
        if "actions" not in response:
            raise KeyError(f"Inference response has no 'actions' field: {response.keys()}")

        actions = np.asarray(response["actions"])
        ground_truth, valid_gt_count = _single_client._load_ground_truth_actions(  # noqa: SLF001
            dataset, frame_index, len(actions)
        )
        errors = np.linalg.norm(actions[:valid_gt_count, :3] - ground_truth[:valid_gt_count, :3], axis=-1)
        position_rmse = float(np.sqrt(np.mean(errors**2)))

        server_ms = float(response.get("server_timing", {}).get("infer_ms", math.nan))
        policy_ms = float(response.get("policy_timing", {}).get("infer_ms", math.nan))
        delay_steps = min(math.ceil(round_trip_ms / 1000 * fps), len(actions))
        remaining_steps = len(actions) - delay_steps
        phase = "warm-up" if request_index < args.warmup_requests else "steady"
        print(
            f"[{request_index + 1}/{args.num_requests}] {phase} frame={frame_index} "
            f"rtt={round_trip_ms:.2f} ms server={server_ms:.2f} ms policy={policy_ms:.2f} ms "
            f"delay_steps={delay_steps}/{len(actions)} remaining={remaining_steps} "
            f"position_rmse={position_rmse:.6f} m"
        )

        round_trip_times.append(round_trip_ms)
        server_times.append(server_ms)
        policy_times.append(policy_ms)
        final_actions = actions
        final_ground_truth = ground_truth
        final_tcp_reference = tcp_reference

        if args.request_interval_s > 0 and request_index + 1 < args.num_requests:
            time.sleep(args.request_interval_s)

    steady_slice = slice(args.warmup_requests, None)
    steady_round_trip = round_trip_times[steady_slice]
    steady_server = [value for value in server_times[steady_slice] if np.isfinite(value)]
    steady_policy = [value for value in policy_times[steady_slice] if np.isfinite(value)]

    print("\nSteady-state latency summary (warm-up excluded):")
    _print_latency_summary("round_trip", steady_round_trip)
    _print_latency_summary("server_infer", steady_server)
    _print_latency_summary("policy_infer", steady_policy)
    p95_delay_steps = math.ceil(_percentile(steady_round_trip, 95) / 1000 * fps)
    print(
        f"P95 latency corresponds to about {p95_delay_steps} action step(s) at {fps:g} Hz; "
        f"the {action_horizon}-step chunk retains about {max(action_horizon - p95_delay_steps, 0)} step(s)."
    )

    if args.visualization_path is not None:
        assert final_actions is not None
        assert final_ground_truth is not None
        assert final_tcp_reference is not None
        _single_client._write_trajectory_visualization(  # noqa: SLF001
            args.visualization_path,
            final_actions,
            final_ground_truth,
            final_tcp_reference,
            pose_stride=4,
            pose_axis_length=0.005,
        )
        print(f"Wrote final-request visualization: {args.visualization_path.resolve()}")


if __name__ == "__main__":
    main(tyro.cli(Args))
