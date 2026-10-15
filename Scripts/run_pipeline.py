#!/usr/bin/env python3
"""
Pipeline completo de UNA máquina. Llama a los scripts existentes, limitando
las particiones a las que asignacion.json le da a la máquina.

    1. descarga        -> download_dataset.py (solo si no existen los datos)
    2. particion       -> partition_data.py --particiones ...
    3. procesamiento   -> process_data_by_client.py --particiones ...
    4. simulacion      -> run_simulations.py --particiones <una> --salida ... --cpus ...
                          (una llamada por partición, procesos en paralelo)
    5. consolidacion   -> junta los CSV de la máquina en reports/<maquina>/

De asignacion.json se toma:
    - "particiones":       qué particiones corre la máquina.
    - "procesos":          cuántas particiones se simulan a la vez.
    - "cpus_por_proceso":  CPUs de cada simulación (--cpus).

Uso (desde la raíz del proyecto o desde Scripts/):
    python Scripts/run_pipeline.py --maquina m2
    python Scripts/run_pipeline.py --maquina m1 --desde simulacion
    python Scripts/run_pipeline.py --maquina m3 --sample      # prueba con data/sample

Salidas:
    data/partitioned/K*/<escenario>_seed*/   solo particiones asignadas
    data/federated/K*/<escenario>_seed*/     solo particiones asignadas
    reports/<maquina>/<K_escenario_seed>/    resultados y run.log de cada partición
    reports/<maquina>/*.csv                  resultados consolidados de la máquina
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common_pipeline import ROOT, get_paths, setup_console  # noqa: E402
import partition_data as pdata                               # noqa: E402
import run_simulations as rsim                               # noqa: E402
import pandas as pd                                          # noqa: E402

ASSIGNMENT = ROOT / "assignment.json"
STAGES = ("descarga", "particion", "procesamiento", "simulacion", "consolidacion")
CSV = ("runs.csv", "rounds.csv", "valores.csv", "clients.csv", "central_reference.csv")
NAME_RE = re.compile(r"^K(\d+)_(.+)_seed(\d+)$")

_lock = threading.Lock()


def log(msg: str) -> None:
    with _lock:
        print(msg, flush=True)

@dataclass
class Machine:
    name: str
    processes: int
    cpus: int
    partitions: list[str] = field(default_factory=list)


def load_assignment() -> dict[str, Machine]:
    data = json.loads(ASSIGNMENT.read_text(encoding="utf-8"))["machines"]
    machines: dict[str, Machine] = {}
    owner: dict[str, str] = {}

    for name, config in data.items():
        m = Machine(name, int(config["processes"]), int(config["cpus_per_process"]))
        for pid in config["partitions"]:
            owner[pid] = name
            m.partitions.append(pid)
        machines[name] = m

    rest = [pid for pid in [f"K{K}_{sid}_seed{seed}" for K in pdata.DEFAULT_KS for sid in pdata.SCENARIOS_ID for seed in pdata.DEFAULT_SEEDS] if pid not in owner]
    if rest:
        print(f"Aviso: particiones del plan sin asignar: {', '.join(rest)}", flush=True)
    return machines


def get_folder(raiz: Path, pid: str) -> Path:
    K, resto = pid.split("_", 1)
    return raiz / K / resto


def run(script: str, args: list[str], log_file: Path | None = None) -> int:
    cmd = [sys.executable, str(HERE / script), *args]
    env = {**os.environ, "PYTHONUTF8": "1", "FLWR_FLCRM": str(ROOT)}
    if log_file is None:
        return rc(cmd, env)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n===== {script} {' '.join(args)} =====\n")
        f.flush()
        return rc(cmd, env, stdout=f)

def rc(cmd: list[str], env: dict, stdout=None) -> int:
    return subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout,
                          stderr=subprocess.STDOUT if stdout else None).returncode


def flag_sample(sample: bool) -> list[str]:
    return ["--sample"] if sample else []

def download_step(paths) -> None:
    if (paths.raw / "application_train.csv").exists():
        log(f"[descarga] Ya existen los datos en {paths.raw}; se omite.")
        return
    log("[descarga] Descargando el dataset con download_dataset.py ...")
    if run_root("download_dataset.py") != 0 or not (paths.raw / "application_train.csv").exists():
        raise SystemExit("[descarga] No se pudo descargar el dataset.")


def run_root(script: str) -> int:
    return subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT).returncode


def partition_step(paths, m: Machine, sample: bool) -> None:
    args = ["--partitions", ",".join(m.partitions), *flag_sample(sample)]
    if run("partition_data.py", args) != 0:
        raise SystemExit("[particion] partition_data.py terminó con error.")


def process_step(paths, m: Machine, sample: bool) -> None:
    args = ["--partitions", ",".join(m.partitions), *flag_sample(sample)]
    if run("process_data_by_client.py", args) != 0:
        raise SystemExit("[procesamiento] process_data_by_client.py terminó con error.")


def simulation_step(paths, m: Machine, sample: bool) -> list[str]:
    listas = [pid for pid in m.partitions
              if (get_folder(paths.federated, pid) / "_PROCESSED").exists()]
    sin = [pid for pid in m.partitions if pid not in listas]
    if sin:
        log(f"[simulacion] Sin datos procesados, se omiten: {', '.join(sin)}")
    if not listas:
        log("[simulacion] No hay particiones procesadas para simular.")
        return []

    log(f"[simulacion] {len(listas)} particiones | {m.processes} en paralelo x {m.cpus} CPUs")

    def work(pid: str) -> tuple[str, int]:
        out = paths.reports / m.name / pid
        out.mkdir(parents=True, exist_ok=True)
        args = ["--partitions", pid, "--out", str(out), "--cpus", str(m.cpus), *flag_sample(sample)]
        log(f"[simulacion] {pid}: empieza")
        rc = run("run_simulations.py", args, log_file=out / "run.log")
        estado = "listo" if rc == 0 else f"FALLÓ (código {rc}); revisa {out / 'run.log'}"
        log(f"[simulacion] {pid}: {estado}")
        return pid, rc

    with ThreadPoolExecutor(max_workers=m.processes) as ex:
        resultados = list(ex.map(work, listas))
    return [pid for pid, rc in resultados if rc != 0]


def consolidate_step(paths, m: Machine) -> None:
    base = paths.reports / m.name
    if not base.exists():
        log("[consolidacion] No hay resultados que consolidar.")
        return

    folders = sorted(d for d in base.iterdir() if d.is_dir())
    for name in CSV:
        parts = [pd.read_csv(d / name) for d in folders if (d / name).exists()]
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(base / name, index=False)

    if (base / "runs.csv").exists():
        rsim.summarize(base)
        log(f"[consolidacion] Resultados de {m.name} en {base}")


def main() -> None:
    setup_console()

    ap = argparse.ArgumentParser(description="Pipeline completo para una máquina según asignacion.json")
    ap.add_argument("--machine", required=True, help="m1, m2, m3 o m4 (según asignacion.json).")
    ap.add_argument("--f", choices=STAGES, default="descarga",
                    help="Etapa desde la que empieza; sigue hasta el final.")
    ap.add_argument("--sample", action="store_true", help="Usa data/sample (prueba rápida).")
    args = ap.parse_args()

    machines = load_assignment()
    if args.machine not in machines:
        raise SystemExit(f"'{args.machine}' no está en asignacion.json. Opciones: {', '.join(machines)}")
    m = machines[args.machine]

    paths = get_paths(args.sample)
    paths.make()
    stages = STAGES[STAGES.index(args.f):]

    log(f"Máquina {m.name}: {len(m.partitions)} particiones | {m.processes} proceso(s) x {m.cpus} CPUs | "
        f"etapas: {', '.join(stages)}")
    if args.sample:
        log("MODO SAMPLE: se usa data/sample")

    if "descarga" in stages and not args.sample:
        download_step(paths)
    if "particion" in stages:
        partition_step(paths, m, args.sample)
    if "procesamiento" in stages:
        process_step(paths, m, args.sample)

    infactible = [pid for pid in m.partitions if not (get_folder(paths.partitioned, pid) / "_COMPLETE").exists()]
    if infactible:
        log(f"Particiones sin generar (no factibles o no hechas): {', '.join(infactible)}")

    fails: list[str] = []
    if "simulacion" in stages:
        fails = simulation_step(paths, m, args.sample)
    if "consolidacion" in stages:
        consolidate_step(paths, m)

    if fails:
        log("Simulaciones con error: " + ", ".join(fails))
        raise SystemExit(1)
    log("Pipeline terminado.")


if __name__ == "__main__":
    main()