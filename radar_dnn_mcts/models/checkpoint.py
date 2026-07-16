from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, modules: dict[str, torch.nn.Module], metadata: dict | None = None) -> None:
    payload = {"format": 1, "metadata": metadata or {}, "modules": {name: module.state_dict() for name, module in modules.items()}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, modules: dict[str, torch.nn.Module], map_location: str | torch.device = "cpu") -> dict:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if payload.get("format") != 1:
        raise ValueError("unsupported checkpoint format")
    missing = set(modules) - set(payload["modules"])
    if missing:
        raise KeyError(f"checkpoint is missing modules: {sorted(missing)}")
    for name, module in modules.items():
        module.load_state_dict(payload["modules"][name])
    return dict(payload.get("metadata", {}))
