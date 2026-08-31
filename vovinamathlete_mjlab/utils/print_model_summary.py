from __future__ import annotations

from typing import Any


def _count_params(module: Any) -> tuple[int, int]:
  total = sum(p.numel() for p in module.parameters())
  trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
  return total, trainable


def print_model_summary(alg: Any) -> None:
  print("\n" + "=" * 88)
  print("[MODEL PARAMETER SUMMARY]")
  print("=" * 88)

  grand_total = 0
  for name in ("actor", "critic", "discriminator"):
    module = getattr(alg, name, None)
    if module is None:
      continue
    total, trainable = _count_params(module)
    grand_total += total
    print(f"  {name:12s} total={total:,}  trainable={trainable:,}")

  print(f"  {'TOTAL':12s} total={grand_total:,}")
  print("=" * 88 + "\n")
