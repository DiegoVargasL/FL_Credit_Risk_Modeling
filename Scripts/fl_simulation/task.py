from __future__ import annotations

import os
from functools import lru_cache
import zlib
import numpy as np
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq

from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

ID, TARGET = "SK_ID_CURR", "TARGET"
META_COLS = (ID, TARGET, "global_split", "local_split")

# Constantes para WoE_logreg
N_BINS = 10
MIN_IV = 0.02
MAX_FEATURES = 150
SMOOTHING = 0.5
LR = 0.3
L2 = 1e-4
BATCH = 256
LOCAL_EPOCHS = 2
MAX_LOCAL_STEPS = 80
NAN_CELL = NCELL = 800
NC1 = NCELL + 1
KMAX = N_BINS + 2
T_LO = -16.0
T_STEP = 0.05

# Constantes para modelos LightGBM y Xgboost
TREES_LOCAL = 3

XGB_PARAMS = dict(objective="binary:logistic",
                  eta=0.1,
                  max_depth=4,
                  min_child_weight=5,
                  subsample=0.8,
                  colsample_bytree=0.5,
                  reg_lambda=1.0,
                  tree_method="hist",
                  base_score=0.5,
                  nthread=1,
                  verbosity=0)

LGBM_PARAMS = dict(objective="binary",
                   learning_rate=0.1,
                   num_leaves=15,
                   min_child_samples=20,
                   subsample=0.8,
                   subsample_freq=1,
                   colsample_bytree=0.5,
                   reg_lambda=1.0,
                   boost_from_average=False,
                   num_threads=1,
                   verbose=-1)

# Constantes para la convergencia del modelo.
MAX_ROUNDS = 200 # Si llega a este numero de rondas y no ha convergido se detiene.
MIN_ROUNDS = 20 # Hasta esta ronda se empieza a evaluar la convergencia.
CONV_WINDOW = 5 # El AUC de valid se suaviza con el promedio de CONV_WINDOW ultimas rondas.
CONV_PATIENCE = 10 # Converge si el AUC suavizado no mejora en estas rondas.
CONV_TOL = 1e-4 # El valor del AUC suavizado debe mejorar por más de CONV_TOL para considerarse una mejora.

# Niveles de heterogeneidad de sistemas:
# (Dispersión del factor de infraestructura, caída maxima de cada cliente)
SYSTEM_LEVELS = {"baja": dict(sigma=0.0, dropout_max=0.0),
                 "media": dict(sigma=0.5, dropout_max=0.10),
                 "alta": dict(sigma=1.0, dropout_max=0.30)}

# Estrategias de selección de clientes.
STRATEGIES = ("random", "greedy", "lp")

# Constantes de presupuestos y costos
BUDGET_FRACTIONS = (0.2, 0.4, 0.6)
FIXED_COST_SHARES = (0.1, 0.3, 0.6)
DEFAULT_BUDGET_FRACTION = 0.4
DEFAULT_FIXED_COST_SHARE = 0.3
REF_VAL = (1.0, 0.0)

# Constantes Modelo de Contribución
Q_PRIOR = 0.5
Q_PRIOR_WEIGHT = 2.0
FORGET = 0.9
UCB_C = 0.3
REL_A = 4.0
REL_B = 1.0

class ContributionModel:
    """
        Modelo de el valor de contribución de cada cliente.
        Se actuacliza cada ronda con lo que observa el modelo.
        v_k = g_k (Prior por tamaño) * r_k (Confiabilidad) * q_k (Calidad aprendida + bono de exploración)
    """

    def __init__(self, n: np.ndarray):
        K = len(n)
        self.g = np.sqrt(n) / np.sqrt(n).mean()
        self.S1, self.S0 = np.zeros(K), np.zeros(K)
        self.att, self.succ = np.zeros(K), np.zeros(K)

    def start_round(self) -> None:
        self.S1 *= FORGET
        self.S0 *= FORGET

    def observe_quality(self, K: int, y_obs: float) -> None:
        self.S1[K] += y_obs
        self.S0[K] += 1.0

    def values(self, t:int) -> np.ndarray:
        q = (Q_PRIOR_WEIGHT * Q_PRIOR + self.S1) / (Q_PRIOR_WEIGHT + self.S0)
        bonus = UCB_C * np.sqrt(2.0 * np.log(1.0 + t) / (self.S0 + 1.0))
        r = (self.succ + REL_A) / (self.att + REL_A + REL_B)
        return self.g * r * np.minimum(1.0, q+bonus)

    def observe_attempts(self, selected, responded) -> None:
        self.att[selected] += 1
        self.succ[responded] += 1


class ConvergenceMonitor():
    """
    """

    def __init__(self):
        self.hist: list[float] = []
        self.best = -np.inf
        self.t_star = 0 

    def update(self, t:int, auc_valid: float) -> bool:
        self.hist.append(auc_valid)
        w = [a for a in self.hist[-CONV_WINDOW:] if not np.isnan(a)]
        
        if not w:
            return False
        
        s = float(np.mean(w))
        if s > self.best + CONV_TOL:
            self.best, self.t_star = s, t
        
        return t >= MIN_ROUNDS  and t - self.t_star >= CONV_PATIENCE




def empty_tables() -> dict:
    return {"sel": np.zeros(0, np.int32), "c2b": np.zeros((0, NC1), np.int16), "woe": np.zeros((0, KMAX), np.float32)}

def build_woe(counts: np.ndarray) -> dict:
    """
        Construye las tablas WOE a partir de los conteos sumados de los clientes vistos.
    """

    counts = counts.astype(np.int64)
    d = counts.shape[0]
    n_good, n_bad = float(counts[0, 0].sum()), float(counts[0, 1].sum())
    if n_good == 0 or n_bad == 0:
        return empty_tables()

    ivs, tabs = np.full(d, -1.0), {}
    for j in range(d):
        c = counts[j, 0, :NCELL] + counts[j, 1, :NCELL]
        N = c.sum()
        if N == 0:
            continue

        cum = np.cumsum(c)
        last = int(np.flatnonzero(c)[-1])
        cuts = np.unique(np.searchsorted(cum, N * np.arange(1, N_BINS) / N_BINS))
        cuts = cuts[cuts < last]
        if cuts.size == 0:
            continue

        c2b = np.searchsorted(cuts, np.arange(NC1)).astype(np.int16)
        c2b[NAN_CELL] = cuts.size + 1
        K = cuts.size + 2
        bad = np.bincount(c2b, weights=counts[j, 1], minlength=K)
        tot = np.bincount(c2b, weights=counts[j, 0] + counts[j,1], minlength=K)
        good = tot - bad

        percentage_bad = (bad + SMOOTHING) / (n_bad + SMOOTHING * K)
        percentage_good = (good + SMOOTHING) / (n_good + SMOOTHING * K)
       
        woe = np.log(percentage_good / percentage_bad)
        woe[tot==0] = 0.0

        ivs[j] = float(np.sum((percentage_good - percentage_bad) * woe))
        w = np.zeros(KMAX, np.float32)
        w[:K] = woe

        tabs[j] = (c2b, w)

    order = [j for j in np.argsort(-ivs) if j in tabs]
    chosen = [j for j in order if ivs[j] >= MIN_IV][:MAX_FEATURES]
    if not chosen:
        return empty_tables()

    return {"sel":np.array(chosen, np.int32),
            "c2b": np.stack([tabs[j][0] for j in chosen]),
            "woe": np.stack([tabs[j][1] for j in chosen])}

def cell_index(x:np.ndarray) -> np.ndarray:
    """
        Regresa la celda para cada valor (NaN -> NAN_CELL y valores fuera del rango -> celda extrema)
    """

    nan = np.isnan(x)
    with np.errstate(invalid="ignore", over="ignore"):
        t = np.sign(x) * np.log1p(np.abs(x))
        idx = np.floor((t - T_LO) / T_STEP)

    idx[nan] = 0
    idx = np.clip(idx, 0, NCELL - 1).astype(np.int16)
    idx[nan] = NAN_CELL

    return idx


def woe_features(x: np.ndarray, tab: dict) -> np.ndarray:
    """
        Regresa la matriz (n, d) con el WOE de las variables seleccionadas y 0 en las demas.
    """

    out = np.zeros(x.shape, np.float32)
    sel = tab["sel"]
    if len(sel) == 0:
        return out
    idx = cell_index(x[:, sel]).astype(np.int64)
    for a, j in enumerate(sel):
        out[:, j] = tab["woe"][a][tab["c2b"][a][idx[:, a]]]

    return out

def _lgb():
    import lightgbm as lgb
    return lgb

def _xgb():
    import xgboost as xgb
    return xgb


def predict_block(lib: str, blob: bytes, x: np.ndarray) ->np.ndarray:
    """
        Suma de arboles (margen) de un bloque.
    """

    if lib == "lightgbm":
        lgb = _lgb()
        b = lgb.Booster(model_str=blob.decode("utf-8"))
        return b.predict(x, raw_score=True).astype(np.float64)

    xgb = _xgb()
    b = xgb.Booster()
    b.load_model(bytearray(blob))
    return b.predict(xgb.DMatrix(x, missing=np.nan), output_margin=True).astype(np.float64)


def gain_importance(lib: str, blob: bytes, d:int) -> np.ndarray:
    """
        Importancia por ganancia de cada variable en un bloque.
        Se usa en el servidor como una señal de calida.
    """

    if lib == "lightgbm":
        lgb = _lgb()
        b = lgb.Booster(model_str=blob.decode("utf-8"))
        return b.feature_importance("gain").astype(np.float64)

    v = np.zeros(d, dtype=np.float64)
    xgb = _xgb()
    b = xgb.Booster()
    b.load_model(bytearray(blob))
    for name, g in b.get_score(importance_type="gain").items():
        v[int(name[1:])] = g

    return v

def abs_path(path: str) -> Path:
    """
        Flower antes de ejecutar la app, copia esta en la carpeta ~/.flwr/app/...
        por lo que las rutas relativas ya no funcionan porque en la corpeta donde se 
        ejecuta la app no es la misma donde estan las particiones y datos, asi que
        hay que usar la ruta absoluta que run_simulations.py agrega como variable de
        entorno.
    """
    q = Path(path)
    if q.is_absolute():
        return q

    return Path(os.environ.get("FLWR_FLCRM") or Path.cwd()).resolve() / q

def client_files(part_dir:str) -> list[Path]:
    return sorted(abs_path(part_dir).glob("client_*.parquet"), key = lambda f: int(f.stem.split("_")[1]))

@lru_cache(maxsize=2) # Esto es para evitar leer la misma información 2 veces el archivo parquet
def feature_names(part_dir:str) -> tuple[str, ...]:

    # Ordena los clientes por su numero asignado al particionar.
    names = pq.ParquetFile(client_files(part_dir)[0]).schema_arrow.names
    return tuple(c for c in names if c not in META_COLS)

def load_pooled(part_dir: str, column: str, value: str) -> tuple[np.ndarray, np.ndarray]:
    """
        Devuelve todas las filas donde 'column' == 'value'
        en todos los clientes de la partición.
    """

    feats, xs, ys = list(feature_names(part_dir)), [], []

    for f in client_files(part_dir):
        df = pd.read_parquet(f)
        df = df[df[column] == value]
        xs.append(df[feats].to_numpy(np.float32))
        ys.append(df[TARGET].to_numpy(np.float32))

    return np.concatenate(xs), np.concatenate(ys)


def local_counts(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n, d, = x.shape
    out = np.zeros((d, 2 * NC1), dtype=np.int64)

    CHUNKSIZE = 100_000

    y = y.astype(int)
    for s in range(0, n, CHUNKSIZE):
        idx = cell_index(x[s:s + CHUNKSIZE]).astype(np.int64)
        off = NC1 * y[s:s+CHUNKSIZE]
        for j in range(d):
            out[j] += np.bincount(idx[:, j] + off, minlength=2 * NC1)

    return out.reshape(d, 2, NC1).astype(np.int32)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def score_metrics(y: np.ndarray, margin: np.ndarray) -> dict:
    """
        Devuelve las metricas AUC-ROC, KS, AUC-PR y log-loss a partir de margenes.
    """

    p = sigmoid(margin)
    pc = np.clip(p, 1e-7, 1- 1e-7)
    out = {"logloss": float(-(y*np.log(pc) + (1-y) * np.log(1 - pc)).mean())}
    
    if len(np.unique(y)) < 2:
        return {"roc_auc": float("nan"), "ks": float("nan"), "pr_auc": float("nan"), **out}

    fpr, tpr, _ = roc_curve(y, margin)

    return {
        "roc_auc": float(roc_auc_score(y, margin)),
        "ks": float(np.max(tpr - fpr)),
        "pr_auc": float(average_precision_score(y, margin)),
        **out
    }


def make_prices(n:np.ndarray, m:np.ndarray, fixed_cost: float) -> np.ndarray:
    """
        Calcula los costos normalizados (media = 1) por cliente.
        c_k = (F + (1-F) n_k / media(n)) * m_k.
    """

    base = fixed_cost + (1-fixed_cost) * n / n.mean()
    price = base * m
    return price / price.mean()

def fill_budget(order, prices, B) -> list[int]:
    """
        Regresa una lista de clientes que se ajuste al presupuesto.
        Le da preferencia a los primero clientes en el orden dado.
    """
    chosen, spent = [], 0.0
    for K in order:
        if spent + prices[K] <= B + 1e-9:
            chosen.append(int(K))
            spent += prices[K]

    return chosen

def select_random(prices: np.ndarray, B: float, rng: np.random.Generator) -> list[int]:
    """
        Regresa la lista de clientes selccionados por un orden al azar.
    """
    return fill_budget(rng.permutation(len(prices)), prices, B)

def select_greedy(v, prices, B) -> list[int]:
    """
        Regresa la lista de clientes seleccionados por un orden
        de mayor a menor razon valor/costo.
    """

    return fill_budget(np.argsort(-(v/prices), kind="stable"), prices, B)

def select_lp(v, prices, B) -> list[int]:
    """
        Regresa la lista de clientes realizando un algoritmo de mochila 0-1 optimo.
        max sum(v_k * x_k)
        s. a.
        sum(prices_k * x_k) <= B

        Donde x_k es una variable binaria 0-1
        (1 si el cliente k es seleccionado, 0 si no)

        Nota: Como milp minimiza, por eso tomamos -v.
    """

    from scipy.optimize import Bounds, LinearConstraint, milp

    res = milp(c=-v, constraints=LinearConstraint(prices[None, :], -np.inf, B), integrality=np.ones(len(v)), bounds=Bounds(0, 1))

    return [] if res.x is None else [int(k) for k in np.flatnonzero(res.x > 0.5)]

@lru_cache(maxsize=32)
def load_client_train(part_dir: str, pid: int) -> tuple[np.ndarray, np.ndarray]:
    """
        Devuelve el set de entrenamiento local del cliente 'pid'.
    """

    df = pd.read_parquet(client_files(part_dir)[pid])
    df = df[df["local_split"] == "train"]
    return df[list(feature_names(part_dir))].to_numpy(np.float32), df[TARGET].to_numpy(np.float32)


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(k) for k in key]))


def infra_factor(level: str, seed: int, pid: int) -> float:
    """
        Devuelve el facto de infraestructura m_k.
    """

    sigma = SYSTEM_LEVELS[level]["sigma"]
    if sigma == 0:
        return 1.0

    return float(_rng(seed, 7001, pid, zlib.crc32(level.encode())).lognormal(-sigma ** 2 / 2, sigma))


def drop_prob(level: str, seed: int, pid: int) -> float:
    """
        Devuelve una probabilidad aleatoria de caida del cliente.
    """

    mx = SYSTEM_LEVELS[level]["dropout_max"]
    return 0.0 if mx == 0 else float(_rng(seed, 7002, pid, zlib.crc32(level.encode())).uniform(0.0, mx))


def drop_draw(seed: int, pid: int, t: int) -> float:
    """
        Devuelve un numero aleatorio que decide si el cliente abandona en la ronda t.
    """

    return float(_rng(seed, 7003, pid, t).random())


def sgd_logreg_update(xw: np.ndarray, y:np.ndarray, w:np.ndarray, b: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """
    """

    n = len(y)
    w = w.astype(np.float32).copy()
    w0, b0 = w.copy(), float(b)
    steps = int(min(MAX_LOCAL_STEPS, np.ceil(LOCAL_EPOCHS * n /  BATCH)))
    m = min(BATCH, n)
    for _ in range(steps):
        idx = rng.integers(0, n, size=m)
        g = sigmoid(xw[idx] @ w + b) - y[idx]
        w -= LR * (xw[idx].T @ g / m + L2 * w)
        b -= LR * float(g.mean())

    return w - w0, b - b0

def ensemble_margin(lib: str, blobs: list[bytes], weights: np.ndarray, x:np.ndarray) -> np.ndarray:
    """
    """

    m = np.zeros(len(x), dtype=np.float64)
    if not blobs:
        return m
    
    for blob, w in zip(blobs, weights):
        m += w * predict_block(lib, blob, x)

    return m


def train_block(lib:str, x: np.ndarray, y:np.ndarray, base_margin: np.ndarray, seed: int) -> bytes:
    """
    """

    if lib == "xgboost":
        xgb = _xgb()
        dm = xgb.DMatrix(x, label=y, base_margin=base_margin, missing=np.nan)
        bst = xgb.train({**XGB_PARAMS, "seed": seed}, dm, num_boost_round= TREES_LOCAL)
        return bytes(bst.save_raw("ubj"))

    lgb = _lgb()
    ds = lgb.Dataset(x, label=y, init_score=base_margin, params={"verbose": -1}, free_raw_data=True)
    bst = lgb.train({**LGBM_PARAMS, "seed": seed}, ds, num_boost_round=TREES_LOCAL)
    return bst.model_to_string().encode("utf-8")