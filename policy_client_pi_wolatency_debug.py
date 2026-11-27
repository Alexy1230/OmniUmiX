"""Sequential client with gripper diagnostics. All original motion options apply."""

import csv
import functools
import json
from pathlib import Path
import tempfile
import threading
import time

import click
import numpy as np
import yaml

import policy_client_pi_wolatency as implementation


class GripperRecorder:
    """Record requests, accepted targets and cached feedback without serial reads."""

    def __init__(self, directory, config):
        self.directory = Path(tempfile.mkdtemp(prefix="run_", dir=directory))
        self.events = (self.directory / "chunks.jsonl").open("w", buffering=1)
        self.feedback_file = (self.directory / "feedback.csv").open("w", buffering=1, newline="")
        self.feedback = csv.writer(self.feedback_file)
        self.feedback.writerow([
            "host_time", "episode", "chunk", "phase", "arm", "width_m",
            "motor_angle_rad", "velocity_rad_s", "torque_nm", "buffer_timestamp",
        ])
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.env = None
        self.episode = 0
        self.chunk = -1
        self.active = False
        self.phase = "idle"
        self.pending_observation = None
        (self.directory / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        self.event("manifest", units={"raw_actions_gripper": "model output units",
                                     "decoded_gripper": "metres", "feedback_width": "metres"},
                   feedback_period_s=0.02, config=config)

    def event(self, name, **values):
        with self.lock:
            row = dict(event=name, host_time=time.time(), episode=self.episode,
                       chunk=self.chunk, phase=self.phase, **values)
            self.events.write(json.dumps(row, default=lambda x: np.asarray(x).tolist(), allow_nan=False) + "\n")

    def measurements(self):
        result = []
        for grip, config in zip(self.env.grippers, self.env.grippers_config):
            custom = config.get("gripper_type", "wsg") == "custom"
            state = grip.get_all_state()
            result.append({
                "width_m": float(grip.get_current_gripper_width()) if custom
                else float(np.asarray(state["gripper_position"])[-1]),
                "motor_angle_rad": float(grip.current_position_rad) if custom else None,
                "velocity_rad_s": float(grip.current_vel) if custom else None,
                "torque_nm": float(grip.current_torque_nm) if custom else None,
                "buffer_timestamp": float(np.asarray(state["gripper_timestamp"])[-1])
                if len(state["gripper_timestamp"]) else None,
            })
        return result

    def attach(self, env):
        self.env = env
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self):
        while not self.stop_event.wait(0.02):
            if not self.active:
                continue
            try:
                measurements = self.measurements()
                with self.lock:
                    for arm, sample in enumerate(measurements):
                        self.feedback.writerow([
                            time.time(), self.episode, self.chunk, self.phase, arm,
                            sample["width_m"], sample["motor_angle_rad"], sample["velocity_rad_s"],
                            sample["torque_nm"], sample["buffer_timestamp"],
                        ])
            except Exception as exc:
                self.event("feedback_error", error=repr(exc))
                self.stop_event.wait(0.1)

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self.events.close()
        self.feedback_file.close()


def instrument(recorder, proprio):
    """Wrap the normal entrypoint; preserve its selection and scheduling."""
    base_env = implementation.BimanualUmiEnv
    base_client = implementation.ProprioPolicyClient if proprio else implementation.PolicyClient
    original_decode = implementation.decode_actions
    original_execute = implementation.execute_action_chunk
    original_observation = implementation.build_observation

    class DebugEnv(base_env):
        def start(self, *args, **kwargs):
            result = super().start(*args, **kwargs)
            recorder.attach(self)
            return result

        def start_episode(self, *args, **kwargs):
            result = super().start_episode(*args, **kwargs)
            recorder.episode += 1
            recorder.chunk = -1
            recorder.active = True
            recorder.phase = "episode"
            recorder.event("episode_start", measured=recorder.measurements())
            return result

        def end_episode(self):
            if recorder.active:
                recorder.event("episode_end", measured=recorder.measurements())
                recorder.active = False
            return super().end_episode()

        def exec_actions(self, actions, timestamps, compensate_latency=False):
            submitted = super().exec_actions(actions, timestamps, compensate_latency)
            recorder.event("command", target_times=timestamps, accepted_times=submitted,
                           target_widths_m=np.asarray(actions)[:, 6::7],
                           compensate_latency=compensate_latency, measured=recorder.measurements())
            return submitted

    class DebugClient(base_client):
        def build_payload(self, images, state):
            payload = super().build_payload(images, state)
            if recorder.active:
                recorder.event("request_payload", payload={
                    key: value for key, value in payload.items() if not key.startswith("observation/image")
                })
            return payload

        def call_server(self, images, state):
            active = recorder.active
            if active:
                recorder.chunk += 1
                recorder.phase = "inference"
                if recorder.pending_observation is not None:
                    recorder.event("observation", **recorder.pending_observation)
                    recorder.pending_observation = None
                recorder.event("request", state=state, image_shapes=[np.shape(image) for image in images],
                               measured=recorder.measurements())
            started = time.time()
            result = super().call_server(images, state)
            if active:
                recorder.event("response", raw_actions=result, latency_s=time.time() - started,
                               measured=recorder.measurements())
            return result

    def observation(obs, *args, **kwargs):
        result = original_observation(obs, *args, **kwargs)
        if recorder.active:
            recorder.pending_observation = {
                "timestamps": obs["timestamp"], "state": result[1], "request_widths_m": result[3],
            }
        return result

    def decode(raw, poses, scales, relative, options):
        result = original_decode(raw, poses, scales, relative, options)
        if recorder.active:
            recorder.event("decoded", decoded_actions=result, inverse_scale=scales,
                           inverse_offset=options.get("model_to_hardware_gripper_offset"),
                           width_limits=options.get("gripper_width_limits"))
        return result

    def execute(env, actions, *args, **kwargs):
        recorder.phase = "playback"
        measured = recorder.measurements()
        recorder.event("chunk_start", selected_actions=actions, measured=measured)
        for arm, sample in enumerate(measured):
            widths = np.asarray(actions)[:, 7 * arm + 6] * 1000
            print(f"[Gripper debug] chunk={recorder.chunk} arm={arm} "
                  f"measured={sample['width_m'] * 1000:.2f}mm "
                  f"first6={widths[:6].round(2).tolist()} last={widths[-1]:.2f}mm "
                  f"first_delta={widths[0] - sample['width_m'] * 1000:+.2f}mm", flush=True)
        try:
            return original_execute(env, actions, *args, **kwargs)
        finally:
            recorder.event("chunk_end", measured=recorder.measurements())
            recorder.phase = "between_chunks"

    implementation.BimanualUmiEnv = DebugEnv
    implementation.PROPRIOCEPTION = proprio
    if proprio:
        implementation.ProprioPolicyClient = DebugClient
    else:
        implementation.PolicyClient = DebugClient
    implementation.build_observation = observation
    implementation.decode_actions = decode
    implementation.execute_action_chunk = execute


original_callback = implementation.main.callback


@functools.wraps(original_callback)
def debug_callback(*args, debug_dir=None, proprio=False, **kwargs):
    root = Path(debug_dir or Path(kwargs["output"]).expanduser() / "gripper_debug").expanduser()
    root.mkdir(parents=True, exist_ok=True)
    with open(Path(kwargs["robot_config"]).expanduser()) as stream:
        config = yaml.safe_load(stream)
    recorder = GripperRecorder(root, config)
    print(f"Gripper diagnostics: {recorder.directory}", flush=True)
    recorder.event("arguments", arguments=kwargs, proprio=proprio)
    instrument(recorder, proprio)
    try:
        return original_callback(*args, **kwargs)
    finally:
        recorder.close()


main = implementation.main
main.callback = debug_callback
main = click.option("--debug-dir", type=click.Path(), help="Debug root; defaults to <output>/gripper_debug")(main)
main = click.option("--proprio", is_flag=True, help="Use a proprioception checkpoint and paired history")(main)


if __name__ == "__main__":
    main()
