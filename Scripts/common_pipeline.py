from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
import argparse

ID = "SK_ID_CURR"
TARGET = "TARGET"
RANDOM_SEED = 42

ROOT = Path(__file__).resolve().parent.parent

def setup_console() -> None:
    """
    Fuerza UTF-8 en la salida estándar.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

@dataclass(frozen = True)
class Paths:
    """ Agrupamos todas las rutas que usamos en el proceso en un solo objeto."""
    root: Path
    raw: Path
    processed: Path
    models: Path
    reports: Path
    figures: Path
    partitioned: Path

    def make(self) -> None:
        for p in (self.processed, self.models, self.reports, self.figures, self.partitioned):
            p.mkdir(parents = True, exist_ok = True)

def get_paths(sample: bool = False) -> Paths:
    base = ROOT / "data" / "sample" if sample else ROOT / "data"
    reports = base / "reports" if sample else ROOT / "reports"
    return Paths(
        root = ROOT,
        raw = base / "raw",
        processed = base / "processed",
        reports = reports,
        models= base / "models",
        figures = reports / "figures",
        partitioned = base / "partitioned",
    )

def add_sample_arg(parser: argparse.ArgumentParser) -> None:
    """Agrega la bandera --sample a un parser de argumentos"""
    parser.add_argument(
        "--sample", action="store_true",
        help="Usa data/sample (subconjunto pequeño) en vez del dataset completo."
    )