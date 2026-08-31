from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MotionFileEntry:
    path: str


def _load_motion_entries_from_yaml(yaml_path: Path) -> list[MotionFileEntry]:
    if not yaml_path.is_file():
        raise AssertionError(f"Invalid YAML path: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file) or {}
    motions = data.get("motions", [])
    base_dir = yaml_path.parent
    entries: list[MotionFileEntry] = []

    for entry in motions:
        if "file" in entry:
            file_path = (base_dir / entry["file"]).resolve()
            if not file_path.is_file():
                raise AssertionError(f"Motion file not found: {file_path}")
            entries.append(MotionFileEntry(str(file_path)))
        elif "folder" in entry:
            folder_path = (base_dir / entry["folder"]).resolve()
            if not folder_path.is_dir():
                raise AssertionError(f"Motion folder not found: {folder_path}")
            npz_files = sorted(p for p in folder_path.iterdir() if p.suffix == ".npz")
            if not npz_files:
                raise AssertionError(f"No .npz files found in folder: {folder_path}")
            entries.extend(
                MotionFileEntry(str(file_path)) for file_path in npz_files
            )
    return entries
