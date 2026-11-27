"""Plot targets and measured feedback from policy_client_pi_wolatency_debug.py."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug-dir", type=Path, required=True, help="The printed run_* directory")
    args = parser.parse_args()
    with (args.debug_dir / "chunks.jsonl").open() as stream:
        events = [json.loads(line) for line in stream]
    with (args.debug_dir / "feedback.csv").open() as stream:
        feedback = list(csv.DictReader(stream))
    episodes = sorted({event["episode"] for event in events if event["event"] == "command"})
    for episode in episodes:
        commands = [event for event in events if event["event"] == "command" and event["episode"] == episode]
        count = len(commands[0]["target_widths_m"][0])
        origin = commands[0]["host_time"]
        fig, axes = plt.subplots(count, 1, figsize=(13, 4 * count), squeeze=False, layout="constrained")
        for arm in range(count):
            times, targets = [], []
            for event in commands:
                for target_time, widths in zip(event["target_times"], event["target_widths_m"], strict=True):
                    if target_time in event["accepted_times"]:
                        times.append(target_time - origin)
                        targets.append(widths[arm] * 1000)
            ax = axes[arm, 0]
            ax.step(times, targets, where="post", label="Accepted width target", color="tab:orange")
            selected = [row for row in feedback if int(row["episode"]) == episode and int(row["arm"]) == arm]
            if selected:
                ax.plot([float(row["host_time"]) - origin for row in selected],
                        [float(row["width_m"]) * 1000 for row in selected],
                        label="Measured width (host sample time)", color="tab:blue")
            for event in events:
                if event["episode"] == episode and event["event"] == "chunk_start":
                    ax.axvline(event["host_time"] - origin, color="gray", alpha=0.4, linewidth=0.8)
            ax.set(title=f"Arm {arm}; vertical lines mark chunk starts", xlabel="Host elapsed time (s)", ylabel="Width (mm)")
            ax.grid(alpha=0.3)
            ax.legend()
        path = args.debug_dir / f"episode_{episode:06d}_gripper.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(path)
    if not episodes:
        print("No submitted action commands were recorded.")


if __name__ == "__main__":
    main()
