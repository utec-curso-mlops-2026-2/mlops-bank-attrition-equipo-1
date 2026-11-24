"""Punto de entrada de la imagen Docker (SageMaker).

- `serve`: lo invoca SageMaker en Batch Transform y en el endpoint online → `bank_attrition.serve`.
- `prepare` / `publish`: pasos Processing del pipeline batch → `bank_attrition.batch`.
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv) or ["serve"]
    if argv[0] == "serve":
        from .serve import main as serve_main

        serve_main()
    elif argv[0] in {"prepare", "publish"}:
        from .batch import main as batch_main

        batch_main(argv)
    else:
        raise SystemExit(f"comando desconocido {argv[0]!r}: use serve | prepare | publish")


if __name__ == "__main__":
    main()
