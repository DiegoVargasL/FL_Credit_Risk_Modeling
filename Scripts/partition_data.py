from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from dataclasses import dataclass
import contextlib


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common_pipeline import (ID, TARGET, RANDOM_SEED, setup_console, add_sample_arg, get_paths)


# CONSTANTES
KEY_COL = "ORGANIZATION_TYPE" # proxy de "institución"
MERGED_LABEL = "OTROS_AGRUPADO" # Categoria que junta las de poco soporte
RARE_THRESHOLD = 0.005 # Criterio propio por explicar
MIN_ROWS = 1_000 # Filas mínimas por cliente
MIN_POS = 100 # Positivos minimos por cliente
MAX_ATTEMPTS = 1_000 # Número máximo de intento
GLOBAL_TEST_SIZE = 0.2 # 80% entrenamiento / 20% validación
LOCAL_VALID_SIZE = 0.2 # 80% entrenamiento / 20% validación
DEFAULT_KS = (10, 5, 20) # K = 10 principal; 5 y 20 de sensibilidad
DEFAULT_SEEDS = (42, 43, 44) # Semillas de partición

# Variables que se utilizan para medir heterogeneidad / auditar
NUMERIC_AUDIT = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_INCOME_TOTAL", "AMT_CREDIT"]
RAW_COLS = [ID, TARGET, KEY_COL, "CODE_GENDER", "DAYS_BIRTH"] + NUMERIC_AUDIT

MAX_ATTEMPTS = 100_000

CHUNKSIZE = 200_000

@dataclass(frozen=True)
class Scenario:
    sid: str
    kind: str
    beta: float = 0.0
    alpha: float = 0.0
    beta_q: float = 0.0
    description: str = ""

SCENARIOS: tuple[Scenario, ...] = (
    Scenario("S0", "IID", description="Control IID: Aleatoria estratificada por TARGET"),
    Scenario("S1_b1.0", "LABEL", beta=1.0, description="Sesgo de etiqueta, p_k ~ Dir(1.0)"),
    Scenario("S1_b0.5", "LABEL", beta=0.5, description="Sesgo de etiqueta, p_k ~ Dir(0.5)"),
    Scenario("S2_b1.0", "QUANTITY", beta=1.0, description="Sesgo de cantidad, p_k ~ Dir(0.5) (Contenido IID)"),
    Scenario("S2_b0.5", "QUANTITY", beta=0.5, description="Sesgo de cantidad, p_k ~ Dir(0.5) (Contenido IID)"),
    Scenario("S3_a1.0", "NATURAL", alpha=1.0, description="Caracteristicas naturales por sector (alpha=1.0, 1=1/K)"),
    Scenario("S3_a0.5", "NATURAL", alpha=0.5, description="Caracteristicas naturales por sector (alpha=0.5, 1=1/K)"),
    Scenario("S4_a0.5_b1.0", "MIXED", alpha = 0.5, beta_q=1.0, description="Mixto (principal): etiqueta alpha=0.5 y tamaños q ~ Dir(1.0)"),
)

SCENARIO_BY_ID = {s.sid: s for s in SCENARIOS}

DATA_LEVELS = (
    ("nula", "S0"),
    ("media", "S1_b1.0"),
    ("alta", "S1_b0.5"),
    ("sectores+tamaños", "S4_a0.5_b1.0"),
    ("tamaños", "S2_b0.5"),
)
LEVEL_OF = {sid: name for name, sid in DATA_LEVELS}
SCENARIOS_ID = tuple(sid for _, sid in DATA_LEVELS)

BUREAU_ID = "SK_ID_BUREAU"

TABLES: list[tuple[str, str, str]] = [
    ("application_train.csv",     ID),
    ("bureau.csv",                ID),
    ("bureau_balance.csv",        BUREAU_ID),
    ("previous_application.csv",  ID),
    ("POS_CASH_balance.csv",      ID),
    ("installments_payments.csv", ID),
    ("credit_card_balance.csv",   ID),
]

def load_application(raw_dir: Path) -> pd.DataFrame:
    f = raw_dir / "application_train.csv"
    df = pd.read_csv(f, usecols=RAW_COLS)
    if df[ID].duplicated().any():
        raise ValueError(f"Hay {ID} duplicados en application_train.csv")

    df[KEY_COL] = df[KEY_COL].fillna("DESCONOCIDA")
    return df

def reproduce_global_split(y: np.ndarray) -> np.ndarray:
    """
        Divide los datos en 80/20 para el entrenamiento.
    """

    idx = np.arange(len(y))
    _, idx_test = train_test_split(idx, test_size=GLOBAL_TEST_SIZE, stratify=y, random_state=RANDOM_SEED)

    is_test = np.zeros(len(y), dtype=bool)
    is_test[idx_test] = True
    return is_test

def build_key_groups(df: pd.DataFrame, is_test: np.ndarray, rare_threshold: float = RARE_THRESHOLD):
    """
    """
    train_key = df.loc[~is_test, KEY_COL]
    share = train_key.value_counts(normalize=True)
    rare = set(share[share < rare_threshold].index)
    key = df[KEY_COL].where(~df[KEY_COL].isin(rare), MERGED_LABEL)

    # Categorias que solo aparecen en la prueba
    unseen = set(df.loc[is_test, KEY_COL]) - set(share.index)
    key = key.where(~df[KEY_COL].isin(unseen), MERGED_LABEL)


    mapping = (df.assign(group=key).groupby([KEY_COL, "group"])[TARGET]
               .agg(rows="size", positive="sum", rate="mean").reset_index()
               .rename(columns={KEY_COL: "ORIGINAL_CAT"}))
    return key, mapping


def group_ids(df: pd.DataFrame, key: pd.Series, scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
    if scenario.kind in ("IID", "QUANTITY", "LABEL"):
        codes = df[TARGET].to_numpy().astype(int)
        names = np.array(["TARGET=0", "TARGET=1"])
    else:
        codes, names = pd.factorize(key, sort=True)
    return np.asarray(codes), np.asarray(names)

def cell_counts(g: np.ndarray, y: np.ndarray, G: int, mask:np.ndarray) -> np.ndarray:
    return np.bincount(g[mask] * 2 + y[mask], minlength=G*2).reshape(G, 2)

def draw_probabilities(scenario: Scenario, K: int, G:int, rng: np.random.Generator) -> np.ndarray | None:
    """
        Devuelve un arreglo con la fracción del grupo que le coresponde a cada cliente.
    """
    if scenario.kind == "IID":
        P = np.full((G, K), 1.0/K)
    elif scenario.kind == "LABEL":
        P = rng.dirichlet(np.full(K, scenario.beta), size=G)
    elif scenario.kind == "QUANTITY":
        q = rng.dirichlet(np.full(K, scenario.beta))
        P = np.tile(q, (G,1))
    elif scenario.kind == "NATURAL":
        P = rng.dirichlet(np.full(K, scenario.alpha), size=G)
    elif scenario.kind == "MIXED":
        q = rng.dirichlet(np.full(K, scenario.beta_q))
        P = rng.dirichlet(scenario.alpha * K * q, size=G)
    else:
        raise ValueError(f"Escenario desconocido: {scenario.kind}")

    if not np.isfinite(P).all() or not np.allclose(P.sum(axis=1), 1.0):
        return None # Dirichlet con alpha muy pequeño puede retornar valores NaN.
    return P


def apportion(n: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
        Ya que todos los clientes reciben el floor(n*P) filas, las filas que 
        sobren se reparten una a una entre los clientes con mayor parte fraccionaria.
    """

    n = np.asarray(n, dtype=np.int64)
    quota = n[:, None] * P
    base = np.floor(quota + 1e-9).astype(np.int64)
    deficit = n - base.sum(axis=1)
    if (deficit < 0).any():
        base = np.floor(quota).astype(np.int64)
        deficit = n - base.sum(axis=1)
    rest = quota - base
    rank = np.argsort(np.argsort(-rest, axis=1, kind="stable"), axis=1, kind="stable")
    return base + (rank < deficit[:, None])

def satisfies_minimum(counts: np.ndarray, min_rows: int, min_pos:int) -> bool:
    """
        Verifica si cada cliente en la partición cumple con el criterio dfe minimo 
        de filas y minimo de filas positivas.
    """

    rows = counts.sum(axis=0)
    pos = counts[1::2].sum(axis=0)
    return bool((rows >= min_rows).all() and (pos >= min_pos).all())


@dataclass
class PartitionResult:
    accepted: bool
    attempts: int
    client: np.ndarray | None
    P: np.ndarray | None

def make_partition(df:pd.DataFrame, key: pd.Series, is_test: np.ndarray, scenario: Scenario, K: int, seed:int,
                   min_rows:int = MIN_ROWS, min_pos:int = MIN_POS, max_attempts: int = MAX_ATTEMPTS) -> PartitionResult:
    """
        Genera una partición con un escenario (scenario), un numero de clientes (K) y una semilla (seed).
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, K, list(SCENARIO_BY_ID).index(scenario.sid)]))
    y = df[TARGET].to_numpy().astype(int)
    g, _ = group_ids(df, key, scenario)
    G = int(g.max()) + 1
    train_mask = ~is_test
    n_train = cell_counts(g, y, G, train_mask).reshape(-1)

    accepted, P, attempts, = False, None, 0
    for attempts in range(1, max_attempts+1):
        P = draw_probabilities(scenario, K, G, rng)
        if P is None:
            continue

        counts = apportion(n_train, np.repeat(P,2, axis=0))
        if satisfies_minimum(counts, min_rows, min_pos):
            accepted = True
            break

    if not accepted:
        return PartitionResult(False, attempts, None, None)

    client = np.full(len(df), -1, dtype=np.int16)
    for mask in (train_mask, is_test):
        n_cells = cell_counts(g, y, G, mask).reshape(-1)
        counts = apportion(n_cells, np.repeat(P, 2, axis=0))
        cell_of_row = g * 2 + y
        for cell in range(G*2):
            rows = np.flatnonzero(mask & (cell_of_row == cell))
            if rows.size == 0:
                continue
            rows = rng.permutation(rows)
            client[rows] = np.repeat(np.arange(K), counts[cell])
    assert (client >= 0).all(), "Quedaron clientes sin filas."
    return PartitionResult(True, attempts, client, P)


def local_split(client: np.ndarray, y:np.ndarray, is_test:np.ndarray, K:int, seed:int, valid_size: float = LOCAL_VALID_SIZE) -> np.ndarray:
    """
        Divide dentro de cada cliente los datos en 80/20 para entrenamiento y validación.
    """

    rng = np.random.default_rng(np.random.SeedSequence([seed, K, 1000]))
    out = np.where(is_test, "test", "train").astype(object)
    for j in range(K):
        for cls in (0,1):
            rows = np.flatnonzero((client == j) & ~is_test & (y == cls))
            if rows.size == 0:
                continue
            rows = rng.permutation(rows)
            n_valid = int(round(valid_size * rows.size))
            out[rows[:n_valid]] = "valid"
    return out.astype(str)


def get_assignment(df, key, is_test, scenario, K, seed, min_rows, min_pos, max_attempts):
    y = df[TARGET].to_numpy().astype(int)
    res = make_partition(df, key, is_test, scenario, K, seed, min_rows, min_pos, max_attempts)
    if not res.accepted:
        return None, f"No fue factible despues de {res.attempts} sorteos."
    a = pd.DataFrame({
        ID: df[ID].to_numpy(),
        "global_split": np.where(is_test, "test", "train"),
        "client": res.client.astype("int8"),
        "local_split": local_split(res.client, y, is_test, K, seed),
    })

    return a, f"generada ({res.attempts -1} rechazos)"


def lookup_clients(index:pd.Index, values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """
        Para cada id devuelve el cliente o -1 si el id nop esta en el mapa.
    """
    pos = index.get_indexer(ids)
    out = np.full(len(ids), -1, dtype=np.int16)
    found = pos >= 0
    out[found] = values[pos[found]]
    return out

def split_table(src: Path, dest_dirs: list[Path], id: str, index: pd.Index, values: np.ndarray, chunksize:int, collect_col: str | None = None):
    """
        Parte un csv en K archivos sin cargarlo por completo en memoria.
    """
    K = len(dest_dirs)
    header = pd.read_csv(src, nrows=0, dtype=str).columns.tolist()
    counts = np.zeros(K, dtype=np.int64)
    total = dropped = 0
    link_ids, link_clients = [], []

    with contextlib.ExitStack() as stack:

        handles = [stack.enter_context(open(d / src.name, "w", encoding="utf-8", newline="")) for d in dest_dirs]
        for h in handles:
            pd.DataFrame(columns=header).to_csv(h, index=False, lineterminator="\n")

        reader = pd.read_csv(src, dtype=str, keep_default_na=False, chunksize=chunksize)
        for chunk in reader:
            ids = pd.to_numeric(chunk[id]).to_numpy(dtype=np.int64)
            client = lookup_clients(index, values, ids)
            total += len(chunk)
            dropped += int((client < 0).sum())
            if collect_col is not None:
                kept = client >= 0
                link_ids.append(pd.to_numeric(chunk[collect_col]).to_numpy(dtype=np.int64)[kept])
                link_clients.append(client[kept])
            for j, part in chunk.groupby(client, sort=True):
                if j< 0:
                    continue
                part.to_csv(handles[j], header=False, index=False, lineterminator="\n")
                counts[j] += len(part)

    assert counts.sum() + dropped == total, f"{src.name}: las filas no cuadran"
    link = (np.concatenate(link_ids), np.concatenate(link_clients)) if collect_col else None
    return total, counts, dropped, link


def build_partition(data_dir: Path, out_dir: Path, assign: pd.DataFrame, K:int, chunksize:int) -> None:
    """
        Hace la partición, crea las carpetas y archivos de los K clientes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_COMPLETE").unlink(missing_ok=True)
    dest_dirs = [out_dir / f"client_{c:02d}" for c in range(K)]
    for d in dest_dirs:
        d.mkdir(exist_ok=True)

    assign.to_parquet(out_dir / "assignment.parquet", index=False)

    for j, g in assign.groupby("client"):
        g[[ID, "global_split", "local_split"]].to_csv(dest_dirs[j] / "split.csv", index=False)

    curr_index = pd.Index(assign[ID].to_numpy(dtype=np.int64))
    curr_values = assign["client"].to_numpy(dtype=np.int16)
    bureau_index = bureau_values = None

    manifest, written = [], {}
    for fname, id in TABLES:
        t0 = time.time()
        if id == ID:
            index, values = curr_index, curr_values
        else:
            index, values = bureau_index, bureau_values
        total, counts, dropped, collected = split_table(data_dir / fname, dest_dirs, id, index, values, chunksize, collect_col = BUREAU_ID if fname == "bureau.csv" else None)

        if fname == "application_train.csv":
            assert dropped == 0, "application_train: filas sin cliente"
            assert np.array_equal(counts, np.bincount(assign["client"], minlength=K)), "application_train: conteos distintos a la asignación"

        if collected is not None:
            b_ids, b_cl = collected
            bureau_index = pd.Index(b_ids)
            bureau_values = b_cl.astype(np.int16)

        written[fname] = counts
        manifest += [{"tabla": fname, "client": j, "filas": int(c)} for j, c in enumerate(counts)]
        manifest.append({"tabla":fname, "client": -1, "filas": int(dropped)})
        print(f"{fname:<28} total {total:>11,} | asignadas {counts.sum():>11,} | excluidas {dropped:>10,} | {time.time() - t0:5.1f}s")

    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)

    (out_dir / "_COMPLETE").write_text("ok\n")

        
def main():
    setup_console()

    parser = argparse.ArgumentParser(description="Partición de datos entre clientes federados")
    add_sample_arg(parser)
    parser.add_argument("--partitions", default=None,
                        help="Lista separada por comas (ej. K5_S0_seed42,K10_S0_seed42). Sin esto, genera todas.")
    args = parser.parse_args()
    wanted = set(args.partitions.split(",")) if args.partitions else None
    args = parser.parse_args()
    paths = get_paths(args.sample)
    paths.make()
    data_dir = paths.raw
    out_root = paths.partitioned

    min_rows = 100 if args.sample else MIN_ROWS
    min_pos = 10 if args.sample else MIN_POS

    # Se ven cuales son las diferentes particiones que se vana realizar
    # y cuales aun no se han hecho.
    plan = []
    for K in DEFAULT_KS:
        for sid in SCENARIOS_ID:
            for seed in DEFAULT_SEEDS:
                out_dir = out_root / f"K{K}" / f"{sid}_seed{seed}"
                plan.append((K, sid, seed, out_dir, (out_dir / "_COMPLETE").exists()))
    pending = [p for p in plan if not p[4] and (wanted is None or f"K{p[0]}_{p[3].name}" in wanted)]

    # Verificar si hay suficiente espacio de almacenamiento para las particiones.
    per_partition = sum((data_dir / f).stat().st_size for f, _ in TABLES)
    need = per_partition * len(pending)
    out_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out_root).free
    print(f"Plan: {len(plan)} particiones ({len(plan) - len(pending)} ya completas, {len(pending)} por hacer)")
    print(f"Espacio estimado: {need / 1e9:.1f} GB ({per_partition / 1e9:.2f} GB por partición) | libre: {free / 1e9:.1f} GB | Salida: {out_root}")
    if need * 1.05 > free:
        raise SystemExit("No hay espacio suiciente.")
    if not pending:
        print("No hay nada que hacer.")
        return

    df = load_application(data_dir)
    y = df[TARGET].to_numpy().astype(int)
    is_test = reproduce_global_split(y)
    key, _ = build_key_groups(df, is_test)
    print(f"Solicitudes: {len(df):,} | entrenamiento global: {(~is_test).sum():,} | prueba global: {is_test.sum():,} | minimos por cliente: {min_rows:,} filas / {min_pos:,} positivos")


    n_ok, infeasible, index_rows = 0, [], []
    for K, sid, seed, out_dir, done in plan:
        tag = f"K={K:<3} {LEVEL_OF[sid]:<6} ({sid:<8}) semilla={seed}"
        row = {"K": K, "nivel": LEVEL_OF[sid], "escenario": sid, "semilla": seed, "path": str(out_dir.relative_to(out_root))}
        index_rows.append(row)

        if wanted is not None and f"K{K}_{out_dir.name}" not in wanted:
            continue

        if done:
            print(f"{tag}: ya completa, se omite.")
            continue
        t0 = time.time()
        assign, how = get_assignment(df, key, is_test, SCENARIO_BY_ID[sid], K, seed,
                                     min_rows, min_pos, MAX_ATTEMPTS)

        if assign is None:
            print(f"{tag}: {how}")
            infeasible.append(tag)
            row["path"] = ""
            continue
        print((f"{tag}: asignación {how}"))
        build_partition(data_dir, out_dir, assign, K, CHUNKSIZE)
        n_ok += 1
        print(f"-> {out_dir} ({time.time() - t0:.1f}s)")

    pd.DataFrame(index_rows).to_csv(out_root / "partitions_index.csv", index=False)
    print(f"Listo: {n_ok}/{len(pending)} particiones escritas en {out_root}")
    if infeasible:
        print(f"Particiones no factibles: " + "; ".join(infeasible))
    
        






if __name__ == "__main__":
    main()