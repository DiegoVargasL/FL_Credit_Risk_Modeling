from __future__ import annotations

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

from common_pipeline import (setup_console, add_sample_arg, get_paths)

LEVEL_NAMES = {
    "S0": "nula",
    "S1_b1.0": "media",
    "S1_b0.5": "alta",
    "S4_a0.5_b1.0": "sectores+tamaños",
    "S2_b0.5": "tamaños",
}

MODELS = ("woe_logreg", "lightgbm", "xgboost")
MAX_ROUNDS = 200
RUNS_PER_PARTITION_MODEL = 10
HERE = Path(__file__).resolve().parent

PASSES = {"base": 10, "extra": 72}
BASE_B, BASE_F = 0.4, 0.3

def summarize(out_dir: Path) -> None:
    f = out_dir / "runs.csv"
    runs = pd.read_csv(f)
    runs = runs[runs["estrategia"] != "all"]

    s = (runs.groupby(["modelo", "K", "heterogeneidad_datos", "sistema", "estrategia", "presupuesto_frac", "costo_fijo"])
         .agg(rondas_conv = ("rondas_conv", "mean"), 
              rondas_conv_std = ("rondas_conv", "std"),
              prop_convergio = ("prop_convergio", "mean"),
              gasto_hasta_conv=("gasto_hasta_conv", "mean"), 
              auc_conv=("auc_conv", "mean"), 
              auc_conv_std=("auc_conv", "std"),
              ratio_conv_vs_central=("ratio_conv_vs_central", "mean"), # Revisar si este aplica
              auc_final=("auc_final", "mean"), 
              auc_std=("auc_final", "std"), 
              ratio_vs_central=("ratio_vs_central", "mean"), # REVISAR
              gasto_total=("gasto_total", "mean"), 
              clientes_alcanzados=("clientes_alcanzados", "mean"),
              gini_participacion=("gini_participacion", "mean"), 
              n_semillas=("semilla", "nunique")).reset_index())
    
    s.to_csv(out_dir / "summary.csv", index=False)

    print(f"Archivo summary.csv creado: {len(s)} filas -> {out_dir / 'summary.csv'}")


def done_rows(runs_f: Path, pid: str, model:str, variants: str) -> int:
    if not runs_f.exists():
        return 0

    r = pd.read_csv(runs_f)
    m= (r["particion"] == pid) & (r["modelo"] == model)
    base = np.isclose(r["presupuesto_frac"], BASE_B) & np.isclose(r["costo_fijo"], BASE_F)
    is_all = r["estrategia"] == "all"

    return int((m & (is_all | base)).sum()) if variants == "base" else ((m & ~is_all & ~base).sum())


def main() -> None:
    setup_console()

    parser = argparse.ArgumentParser(description="Simulaciones de FL bajo presupuesto con Flower")
    add_sample_arg(parser)
    args = parser.parse_args()

    paths = get_paths(args.sample)
    paths.make()

    out_dir = paths.reports
    part_dir = paths.federated
    params = paths.root / "model_params.json"

    idx = paths.partitioned / "partitions_index.csv"
    t = pd.read_csv(idx)
    levels = {(int(r.K), r.escenario, int(r.semilla)): r.nivel for r in t.itertuples()}

    parts = []
    for kdir in  sorted(part_dir.glob("K*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 0):
        if kdir.is_dir() and kdir.name[1:].isdigit():
            parts += [(int(kdir.name[1:]), d) for d in sorted(kdir.iterdir()) if d.is_dir() and (d / "_PROCESSED").exists()]

    flwr = shutil.which("flwr")
    runs_f = out_dir / "runs.csv"

    total = len(parts) * len(MODELS) * len(PASSES)
    step = 0

    for variants, expected in PASSES.items():

        for K, d in parts:
            scenario, seed = d.name.rsplit("_seed", 1)
            level = levels.get((K, scenario, int(seed)), LEVEL_NAMES.get(scenario, scenario))

            for model in MODELS:
                step += 1
                if done_rows(runs_f, f"K{K}_{d.name}", model, variants) >= expected:
                    print(f"[{step}/{total}] Se omitio {variants} K={K} {d.name} {model} porque ya existe el registro.")
                    continue

                run_configuration = (f"""part-dir='{part_dir / d}'
                                        out-dir='{out_dir}'
                                        model='{model}'
                                        K={K}
                                        scenario='{scenario}'
                                        level='{level}'
                                        seed={seed}
                                        rounds={MAX_ROUNDS}
                                        variants='{variants}'
                                        model-params='{params}'
                                    """)

                cmd = [flwr, "run", str(HERE), "--stream", "--run-config", run_configuration, "--federation-config", f"num-supernodes={K} client-resources-num-cpus=1"]
                print(f"[{step}/{total}] {variants} K={K} {d.name} {model}")
                env = {**os.environ, "FLWR_FLCRM": str(paths.root)}
                if subprocess.run(cmd, cwd=paths.root, env=env).returncode != 0:
                    sys.exit(f"La ejecución fallo: {variants} {d.name} {model}")
        
        summarize(out_dir)
    
    summarize(out_dir)
    print("Listo. Simulaciones terminadas.")

if __name__ == "__main__":
    main()