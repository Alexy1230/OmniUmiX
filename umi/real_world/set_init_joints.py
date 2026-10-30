#!/usr/bin/env python3
"""Read current Franka joint positions and write them to a robot config."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import List, Mapping, Sequence, Tuple

import yaml


DEFAULT_FRANKA_PORT = 4242
JOINT_COUNT = 7


def _collection_end(text: str, start: int, opening: str, closing: str) -> int:
    """Return the inclusive end of a bracketed value, ignoring strings/comments."""
    if start >= len(text) or text[start] != opening:
        raise ValueError(f"Expected {opening!r} at offset {start}")

    depth = 0
    quote = None
    escaped = False
    in_comment = False

    for index in range(start, len(text)):
        char = text[index]
        if in_comment:
            if char in "\r\n":
                in_comment = False
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\" and quote == '"':
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char == "#":
            in_comment = True
        elif char in "\"'":
            quote = char
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index

    raise ValueError(f"Unclosed {opening!r} starting at offset {start}")


def _robot_object_spans(text: str) -> List[Tuple[int, int]]:
    robots_match = re.search(
        r'(?m)^[ \t]*(?:"robots"|\'robots\'|robots)[ \t]*:[ \t]*(\[)',
        text,
    )
    if robots_match is None:
        raise ValueError("Config does not contain a flow-style 'robots' list")

    array_start = robots_match.start(1)
    array_end = _collection_end(text, array_start, "[", "]")
    spans: List[Tuple[int, int]] = []
    quote = None
    escaped = False
    in_comment = False
    square_depth = 1
    curly_depth = 0
    object_start = None

    for index in range(array_start + 1, array_end):
        char = text[index]
        if in_comment:
            if char in "\r\n":
                in_comment = False
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\" and quote == '"':
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char == "#":
            in_comment = True
        elif char in "\"'":
            quote = char
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth -= 1
        elif char == "{":
            if square_depth == 1 and curly_depth == 0:
                object_start = index
            curly_depth += 1
        elif char == "}":
            curly_depth -= 1
            if curly_depth < 0:
                raise ValueError("Unbalanced braces in 'robots' list")
            if curly_depth == 0 and object_start is not None:
                spans.append((object_start, index + 1))
                object_start = None

    if curly_depth != 0:
        raise ValueError("Unclosed robot object in 'robots' list")
    return spans


def _last_significant_offset(text: str, end: int) -> int:
    quote = None
    escaped = False
    in_comment = False
    last = -1

    for index, char in enumerate(text[:end]):
        if in_comment:
            if char in "\r\n":
                in_comment = False
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\" and quote == '"':
                escaped = True
            elif char == quote:
                quote = None
                last = index
            continue
        if char == "#":
            in_comment = True
        elif char in "\"'":
            quote = char
        elif not char.isspace():
            last = index

    return last


def _set_joints_in_object(robot_text: str, joints: Sequence[float]) -> str:
    value = json.dumps([float(joint) for joint in joints])
    key_match = re.search(
        r'(?m)(?:^[ \t]*|(?<=[{,])[ \t]*)'
        r'(?:"joints_init"|\'joints_init\'|joints_init)'
        r'[ \t]*:[ \t]*',
        robot_text,
    )
    if key_match is not None:
        value_start = key_match.end()
        while value_start < len(robot_text) and robot_text[value_start].isspace():
            value_start += 1
        if value_start >= len(robot_text) or robot_text[value_start] != "[":
            raise ValueError("'joints_init' must be a list")
        value_end = _collection_end(robot_text, value_start, "[", "]")
        return robot_text[:value_start] + value + robot_text[value_end + 1 :]

    close_offset = robot_text.rfind("}")
    close_line_start = robot_text.rfind("\n", 0, close_offset) + 1
    indent_match = re.search(
        r'(?m)^(?P<indent>[ \t]+)(?:"[^"\n]+"|\'[^\'\n]+\'|[A-Za-z_][\w-]*)[ \t]*:',
        robot_text,
    )
    last_offset = _last_significant_offset(robot_text, close_offset)
    if last_offset < 0:
        raise ValueError("Cannot add 'joints_init' to an empty robot config")

    comma = "" if robot_text[last_offset] == "," else ","
    if indent_match is None:
        return (
            robot_text[: last_offset + 1]
            + comma
            + f' "joints_init": {value}'
            + robot_text[last_offset + 1 :]
        )

    indent = indent_match.group("indent")
    before_close = robot_text[: last_offset + 1] + comma + robot_text[last_offset + 1 : close_line_start]
    return before_close + f'{indent}"joints_init": {value}\n' + robot_text[close_line_start:]


def update_config_text(
    text: str,
    robots: Sequence[Mapping[str, object]],
    joints_by_index: Mapping[int, Sequence[float]],
) -> str:
    spans = _robot_object_spans(text)
    if len(spans) != len(robots):
        raise ValueError(
            "Could not safely match parsed robots to their locations in the config "
            f"({len(robots)} parsed, {len(spans)} located)"
        )

    updated = text
    for robot_index in sorted(joints_by_index, reverse=True):
        start, end = spans[robot_index]
        replacement = _set_joints_in_object(updated[start:end], joints_by_index[robot_index])
        updated = updated[:start] + replacement + updated[end:]
    return updated


def _read_franka_joints(robot_index: int, robot_config: Mapping[str, object]) -> List[float]:
    import zerorpc

    try:
        robot_ip = str(robot_config["robot_ip"])
    except KeyError as exc:
        raise ValueError(f"robots[{robot_index}] is missing 'robot_ip'") from exc
    robot_port = int(robot_config.get("robot_port", DEFAULT_FRANKA_PORT))

    print(f"Reading robots[{robot_index}] from {robot_ip}:{robot_port} ...")
    client = zerorpc.Client(heartbeat=20)
    client.connect(f"tcp://{robot_ip}:{robot_port}")
    try:
        joints = [float(value) for value in client.get_joint_positions()]
    finally:
        client.close()

    if len(joints) != JOINT_COUNT:
        raise ValueError(
            f"robots[{robot_index}] returned {len(joints)} joints; expected {JOINT_COUNT}"
        )
    if not all(math.isfinite(value) for value in joints):
        raise ValueError(f"robots[{robot_index}] returned non-finite joint positions")
    print(f"robots[{robot_index}] joints_init = {json.dumps(joints)}")
    return joints


def _atomic_write(path: Path, text: str) -> None:
    stat_result = path.stat()
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(text)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.chmod(temporary_path, stat_result.st_mode)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", nargs="?", metavar="CONFIG", help="Robot YAML config path")
    parser.add_argument(
        "--config", dest="config_option", metavar="CONFIG", help="Robot YAML config path"
    )
    args = parser.parse_args(argv)
    if args.config_path and args.config_option:
        parser.error("provide the config path either positionally or with --config, not both")
    config = args.config_option or args.config_path
    if not config:
        parser.error("a config path is required")
    return Path(config).expanduser()


def main(argv=None) -> int:
    config_path = _parse_args(argv)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8", newline="") as config_file:
        original_text = config_file.read()
    config = yaml.safe_load(original_text)
    if not isinstance(config, dict) or not isinstance(config.get("robots"), list):
        raise ValueError("Config must contain a 'robots' list")

    robots = config["robots"]
    spans = _robot_object_spans(original_text)
    if len(spans) != len(robots):
        raise ValueError(
            "Could not safely match parsed robots to their locations in the config "
            f"({len(robots)} parsed, {len(spans)} located)"
        )

    joints_by_index = {}
    for index, robot_config in enumerate(robots):
        if not isinstance(robot_config, dict):
            raise ValueError(f"robots[{index}] must be a mapping")
        robot_type = str(robot_config.get("robot_type", ""))
        if robot_type.startswith("franka"):
            joints_by_index[index] = _read_franka_joints(index, robot_config)
        else:
            print(f"Skipping robots[{index}] with robot_type={robot_type!r}")

    if not joints_by_index:
        raise ValueError("Config does not contain any Franka robots")

    updated_text = update_config_text(original_text, robots, joints_by_index)
    _atomic_write(config_path, updated_text)
    print(f"Updated {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
