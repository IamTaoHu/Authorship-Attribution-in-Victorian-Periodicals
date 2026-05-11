"""Generate Phase 5 LoRA rank ablation configs and commands."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config


RANKS = [4, 8, 16, 32]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs" / "phase5" / "ablations"
DEFAULT_COMMAND_PATH = PROJECT_ROOT / "outputs" / "phase5" / "run_commands" / "phase5_lora_ablation_commands.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or run Phase 5 LoRA rank ablation commands.")
    parser.add_argument("--base_config", required=True, type=Path)
    parser.add_argument("--train_file", default="outputs/phase5/data/train_instruction.jsonl")
    parser.add_argument("--eval_file", default="outputs/phase5/data/test_instruction.jsonl")
    parser.add_argument("--config_dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--commands_path", type=Path, default=DEFAULT_COMMAND_PATH)
    parser.add_argument("--run", action="store_true", help="Actually launch generated training commands.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config_path = resolve_repo_path(args.base_config)
    base = load_config(str(base_config_path))
    config_dir = resolve_repo_path(args.config_dir)
    commands_path = resolve_repo_path(args.commands_path)
    config_dir.mkdir(parents=True, exist_ok=True)
    commands_path.parent.mkdir(parents=True, exist_ok=True)

    commands = []
    for rank in RANKS:
        config = build_rank_config(base, rank)
        output_path = config_dir / f"{config['experiment_name']}.yaml"
        output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        commands.append(
            "python src/training/train_decoder_qlora.py "
            f"--config {relative(output_path)} "
            f"--train_file {args.train_file} "
            f"--eval_file {args.eval_file}"
        )
    commands_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"Wrote ablation configs to {config_dir}")
    print(f"Wrote command list to {commands_path}")

    if args.run:
        for command in commands:
            print(f"Running: {command}")
            subprocess.run(command.split(), cwd=PROJECT_ROOT, check=True)
    else:
        print("Training was not launched. Pass --run to execute the generated commands.")


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def build_rank_config(base: dict[str, Any], rank: int) -> dict[str, Any]:
    config = copy.deepcopy(base)
    short_name = str(config.get("model_short_name") or "decoder")
    seed = int(config.get("seed", 42))
    config["lora_r"] = int(rank)
    config["lora_alpha"] = max(2 * int(rank), int(config.get("lora_alpha", 2 * rank)))
    config["experiment_name"] = f"{short_name}_qlora_r{rank}_seed{seed}"
    return config


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
