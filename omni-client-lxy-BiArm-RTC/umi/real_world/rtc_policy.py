"""Timestamped RTC plans and transport. No robot/camera imports."""

import heapq
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from umi.real_world.bimanual_policy import PolicyClient, unpack_actions
from umi.real_world.proprio_policy_client import ProprioPolicyClient


def soft_weights(horizon, frozen, overlap):
    """RTC Eq. 5: full committed prefix, decaying overlap, unguided new tail."""
    if not 0 <= frozen <= overlap <= horizon:
        raise ValueError("Require 0 <= frozen <= overlap <= horizon")
    weights = np.zeros(horizon, np.float32)
    weights[:frozen] = 1
    indices = np.arange(frozen, overlap)
    c = (overlap - indices) / (overlap - frozen + 1)
    weights[indices] = c * np.expm1(c) / np.expm1(1.0)
    return weights


def encode_absolute(actions, scales, options):
    """Robot base xyz/rotvec/metres -> wire absolute pose9/model radians."""
    actions = np.asarray(actions, dtype=float)
    count = len(scales)
    if (
        actions.ndim != 2
        or actions.shape[1] != 7 * count
        or not np.isfinite(actions).all()
    ):
        raise ValueError("Invalid absolute execution actions")
    output = np.empty((len(actions), 10 * count), np.float32)
    offsets = options.get("hardware_to_model_gripper_offset", np.zeros(count))
    for i in range(count):
        block = actions[:, 7 * i : 7 * i + 7]
        r = Rotation.from_rotvec(block[:, 3:6]).as_matrix()
        output[:, 10 * i : 10 * i + 3] = block[:, :3]
        output[:, 10 * i + 3 : 10 * i + 9] = np.concatenate(
            [r[:, :, 0], r[:, :, 1]], axis=-1
        )
        output[:, 10 * i + 9] = block[:, 6] * scales[i] + offsets[i]
    return output


class ActionPlan:
    def __init__(self, timestamps, actions):
        self.times = np.asarray(timestamps, dtype=float).copy()
        self.actions = np.asarray(actions, dtype=float).copy()
        if (
            self.times.ndim != 1
            or len(self.times) < 1
            or self.actions.ndim != 2
            or len(self.times) != len(self.actions)
            or self.actions.shape[1] not in (7, 14)
            or not np.isfinite(self.times).all()
            or not np.isfinite(self.actions).all()
            or np.any(np.diff(self.times) <= 0)
        ):
            raise ValueError(
                "Plan must have finite, strictly increasing timestamps and [T,7*N] actions"
            )

    def sample(self, timestamps):
        query = np.clip(
            np.asarray(timestamps, dtype=float), self.times[0], self.times[-1]
        )
        result = np.empty((len(query), self.actions.shape[1]))
        for start in range(0, self.actions.shape[1], 7):
            for dim in (start, start + 1, start + 2, start + 6):
                result[:, dim] = np.interp(query, self.times, self.actions[:, dim])
            if len(self.times) == 1:
                result[:, start + 3 : start + 6] = self.actions[
                    0, start + 3 : start + 6
                ]
            else:
                result[:, start + 3 : start + 6] = Slerp(
                    self.times,
                    Rotation.from_rotvec(self.actions[:, start + 3 : start + 6]),
                )(query).as_rotvec()
        return result

    def replace_future(self, times, actions, switch, committed_until):
        if switch <= committed_until:
            raise ValueError("Cannot replace already committed targets")
        times, actions = np.asarray(times), np.asarray(actions)
        keep_new = times >= switch
        if not np.any(keep_new):
            raise ValueError("No future actions remain")
        keep_old = self.times < switch
        # Retain a point at the committed boundary: later interpolation must not
        # retroactively alter the trajectory used for the guidance prefix.
        old_times = self.times[keep_old]
        old_actions = self.actions[keep_old]
        if (
            self.times[0] <= committed_until < switch
            and committed_until not in old_times
        ):
            old_times = np.r_[old_times, committed_until]
            old_actions = np.vstack([old_actions, self.sample([committed_until])])
            order = np.argsort(old_times)
            old_times, old_actions = old_times[order], old_actions[order]
        merged = ActionPlan(
            np.r_[old_times, times[keep_new]],
            np.vstack([old_actions, actions[keep_new]]),
        )
        # Bound retained history while keeping a point before the request grid.
        first = max(0, np.searchsorted(merged.times, times[0]) - 1)
        self.times, self.actions = merged.times[first:], merged.actions[first:]


def make_request_plan(
    plan,
    observation_time,
    now,
    dt,
    horizon,
    delay_s,
    lead_s,
    committed_until,
    scales,
    options,
):
    times = observation_time + np.arange(horizon) * dt
    deadline = max(now + delay_s + lead_s, committed_until + dt)
    frozen = int(np.searchsorted(times, deadline, side="right"))
    if frozen >= horizon:
        raise TimeoutError(
            "Observation age + inference allowance consumes the prediction horizon"
        )
    overlap = max(
        frozen, min(horizon, int(np.searchsorted(times, plan.times[-1], side="right")))
    )
    return (
        times,
        frozen,
        {
            "version": 1,
            "prefix_absolute": encode_absolute(plan.sample(times), scales, options),
            "weights": soft_weights(horizon, frozen, overlap),
        },
    )


def response_on_time(times, frozen, now, lead_s, committed_until):
    return bool(times[frozen] > max(now + lead_s, committed_until))


class RTCPolicyClient(PolicyClient):
    proprioception = False

    def call_rtc(self, images, state, rtc):
        payload = self.build_payload(images, state)
        if payload.get("prompt") is None:
            payload.pop("prompt", None)
        payload["rtc"] = rtc
        self.ws.send(self.packer.pack(payload))
        message = self.ws.recv(timeout=self.timeout)
        if isinstance(message, str):
            raise RuntimeError(message)
        response = self.codec.unpackb(message)
        if response.get("rtc_version") != 1:
            raise RuntimeError(
                "Server did not acknowledge RTC; refusing unguided fallback"
            )
        raw = np.asarray(response["actions"], dtype=np.float32)
        unpack_actions(raw, self.num_robots)
        if len(raw) != int(self.metadata["action_horizon"]):
            raise ValueError("RTC response horizon changed")
        return raw


class RTCProprioPolicyClient(ProprioPolicyClient, RTCPolicyClient):
    """RTC transport with the explicit UMI proprioception request contract."""

    proprioception = True


class RTCExecutor:
    """Submit each robot target once; dispatch custom grippers at their due time.

    Run dispatch from the control loop, including while an inference future is
    pending. Robot scheduling still uses the existing interpolation workers.
    """

    def __init__(self, env, debug=False):
        self.env, self.debug = env, debug
        self.pending = []
        self.last_target = -np.inf

    def submit(self, action, target_time, now):
        if target_time <= self.last_target:
            raise ValueError("RTC targets must be strictly increasing")
        if any(
            target_time - float(rc.get("robot_action_latency", 0)) <= now
            for rc in self.env.robots_config
        ):
            raise TimeoutError("RTC robot submission deadline missed")
        action = np.asarray(action, dtype=float)
        if action.shape != (7 * len(self.env.robots),) or not np.isfinite(action).all():
            raise ValueError("Invalid RTC execution action")
        if any(
            target_time - float(gc.get("gripper_action_latency", 0)) <= now
            for gc in self.env.grippers_config
        ):
            raise TimeoutError("RTC gripper submission deadline missed")
        self.last_target = target_time
        if self.debug:
            return
        for i, (robot, gripper, rc, gc) in enumerate(
            zip(
                self.env.robots,
                self.env.grippers,
                self.env.robots_config,
                self.env.grippers_config,
            )
        ):
            robot.schedule_waypoint(
                pose=action[7 * i : 7 * i + 6],
                target_time=target_time - float(rc.get("robot_action_latency", 0)),
            )
            width = float(action[7 * i + 6])
            due = target_time - float(gc.get("gripper_action_latency", 0))
            if gc.get("gripper_type", "wsg") == "custom":
                heapq.heappush(
                    self.pending, (due, i, width, float(gc.get("force_limit_nm", 1.5)))
                )
            else:
                gripper.schedule_waypoint(pos=width, target_time=due)
        if self.env.action_accumulator is not None:
            self.env.action_accumulator.put(
                np.asarray(action)[None], np.array([target_time])
            )

    def dispatch(self, now):
        due_by_arm = {}
        while self.pending and self.pending[0][0] <= now:
            _, i, width, force = heapq.heappop(self.pending)
            due_by_arm[i] = (width, force)
        # If the loop stalls, do not burst old motor commands into its serial queue.
        for i, (width, force) in due_by_arm.items():
            self.env.grippers[i].move_gripper(width, force_limit_nm=force)

    def drain(self):
        # Only the short committed horizon remains, never a full model chunk.
        while time.time() < self.last_target:
            self.dispatch(time.time())
            time.sleep(0.005)
        self.dispatch(time.time())
