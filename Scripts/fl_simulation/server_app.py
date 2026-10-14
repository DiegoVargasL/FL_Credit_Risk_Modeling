from __future__ import annotations

import zlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from flwr.serverapp import ServerApp, Grid
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MessageType, RecordDict

from task import (SYSTEM_LEVELS, STRATEGIES, BUDGET_FRACTIONS, FIXED_COST_SHARES, DEFAULT_BUDGET_FRACTION, DEFAULT_FIXED_COST_SHARE, REF_VAL,
                ContributionModel, ConvergenceMonitor, empty_tables, build_woe, woe_features, predict_block, gain_importance, abs_path, load_pooled,
                local_counts, score_metrics, make_prices, feature_names, select_random, select_greedy, select_lp)

app = ServerApp()


class WoeServer:
    """
        woe_logreg: coeficientes + conteos WOE acumulados de los clientes que ya han participado.
    """

    def __init__(self, d: int):
        self.d = d
        self.coef = np.zeros(d + 1, dtype=np.float32)
        self.counts = None
        self.seen: set[int] = set()
        self.tab = empty_tables()
        self.dirty = False
        self.sets: dict[str, np.ndarray] = {}

    def start_round(self) -> None:
        if self.dirty:
            self.tab = build_woe(self.counts)
            self.dirty = False

    def need_woe(self, K:int) -> int:
        return int(K not in self.seen)

    def start(self, K: int) -> int:
        return 0

    def payload(self, K:int) -> list[np.ndarray]:
        return [self.coef, self.tab["sel", self.tab["c2b"], self.tab["woe"]]]

    def absorb(self, KS:list[int], replies:dict, n: np.ndarray) -> list[np.ndarray]:
        d = [replies[K].content["arrays"].to_numpy_ndarrays()[0] for K in KS]
        w = n[KS] / n[KS].sum()

        self.coef = self.coef + sum(wi * di for wi, di, in zip(w, d)).astype(np.int32)

        for K in KS:
            if "counts" in replies[K].content:
                c = replies[K].content["counts"].to_numpy_ndarrays()[0].astype(np.int64)
                self.counts = c if self.counts is None else self.counts + c
                self.seen.add(K)
                self.dirty = True

        return d

    def bind(self, name:str, x: np.ndarray) -> None:
        self.sets[name] = x

    def margin(self, name:str) -> np.ndarray:
        return woe_features(self.sets[name], self.tab) @ self.coef[:1] + self.coef[-1]

    
class TreeServer:
    """
        xgboost o lightgbm:
        Lista de bloques de arboles con sus pesos.
        El margen de prueba que se acumula en cada ronda.
    """

    def __init__(self, lib: str, d:int):
        self.lib, self.d = lib, d
        self.blobs: list[bytes]= []
        self.weights: list[float] = []
        self.sets: dict[str, np.ndarray] = {}
        self.margins: dict[str, np.ndarray] = {}
        self.synced: dict[int, int] = {}

    def bind(self, name: str, x: np.ndarray) -> None:
        self.sets[name] = x
        self.margins[name] = np.zeros(len(x), dtype=np.float64)

    def start_round(self) -> None:
        pass

    def need_woe(self, K:int) -> int:
        return 0

    def start(self, K:int) -> int:
        return self.synced.get(K, 0)

    def payload(self, K:int) -> list[np.ndarray]:
        s0 = self.start(K)
        return[np.asarray(self.weights[s0:], dtype=np.float64)] + [np.frombuffer(b, dtype=np.uint8) for b in self.blobs[s0:]]

    def absorb(self, KS: list[int], replies:dict, n:np.ndarray) -> list[np.ndarray]:
        for K in KS:
            self.synced[K] = len(self.blobs)

        w = n[KS] / n[KS].sum()
        
        vecs = []
        for wi, K in zip(w, KS):
            blob = replies[K].content["arrays"].to_numpy_ndarrays()[0].tobytes()
            self.blobs.append(blob)
            self.weights.append(float(wi))

            for nm, x, in self.sets.items():
                self.margin[nm] += wi * predict_block(self.lib, blob, x)

            imp = gain_importance(self.lib, blob, self.d)
            vecs.append(imp / imp.sum() if imp.sum > 0 else imp)

        return vecs

    def margin(self, name:str) -> np.ndarray:
        return self.margins[name]



def central_baseline(model: str, config: dict, xtest:np.ndarray, ytest: np.ndarray, out_dir:Path) -> float:
    """
        Entrena el modelo de forma centralizada utilizando solo las filas que los clientes
        utilizan para entrenar el modelo, de esta manera obtenemos un modelo centralizado con
        entrenado con los mismo hiperparametros y los mismos datos, para poder comparar que
        tanto se pierder al federar el entrenamiento.

        Si el modelo ya se entreno con esa partición solo devuelve el AUC que se obtuvo al evaluarlo
        por primera vez.
    """
    
    f = out_dir / "central_reference.csv"
    pid = f"K{int(config['K'])}_{Path(config['part-dir']).name}"
    if f.exists():
        df = pd.read_csv(f)
        if (df["modelo"] == model & df["particion"] == pid).any():
            return float(df.loc[df["modelo"] == model & df["particion"] == pid, "auc"].iloc[0])

    xtr, ytr = load_pooled(config["part-dir"], "local_split", "train")
    params = json.loads(Path(config["model-params"]).read_text(encoding="utf-8"))

    if model == "woe_logreg":
        from sklearn.linear_model import LogisticRegression
        tab = build_woe(local_counts(xtr, ytr))
        sel = tab["sel"]
        
        C = float(params.get("woe_logreg", {}).get("C", 1.0))
        lr = LogisticRegression(C=C, solver="lbfgs", max_iter=1000).fit(woe_features(xtr, tab)[:, sel], ytr)
        margin = lr.decision_function(woe_features(xtr, tab)[:, sel])

    elif model == "xgboost":
        import xgboost as xgb

        clf =xgb.XGBClassifier(
            objective="binary:logistic",
            learning_rate=0.03,
            max_depth = 5,
            min_child_weight = 10,
            subsamble=0.8,
            colsample_bytree=0.5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_metho="hist",
            n_jobs=-1,
            random_state = 42,
            n_estimators = int(params.get("xgboost", {}).get("n_estimators", 300)),
        )

        clf.fit(xtr, ytr)
        margin = clf.predict(xtest, output_margin=True)

    else:
        import lightgbm as lgb

        clf = lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.03,
            num_leaves = 32,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state = 42,
            verbose= -1,
            n_estimators = int(params.get("xgboost", {}).get("n_estimators", 300)),
        )

        clf.fit(xtr, ytr)
        margin = clf.predict(xtest, raw_score=True)

    metrics = score_metrics(ytest, np.asarray(margin, dtype=np.float64))

    pd.concat([pd.read_csv(f) if f.exists() else pd.DataFrame(), pd.DataFrame([{"modelo": model, "particion": pid, **metrics}])]).to_csv(f, index=False)

    print(f"Modelo base centralizado [{model}] {pid}: AUC = {metrics['roc_auc']:.4f}")

    return metrics["roc_auc"]


def query_nodes(grid: Grid, config: dict, system: str) -> list[dict]:
    """
        Obtiene n_k (Filas de entrenamiento que tiene) y m_k (el factor de infraestructura, un log normal 
        de media 1, que depende de el nivel de heterogeneidad de sistema) de cada cliente.
    """

    node_ids = list(grid.get_node_ids())
    content = RecordDict({"config": ConfigRecord({"part-dir": config["part-dir"], "system": system, "seed": int[config["seed"]]})})
    msgs = [Message(content=content, dst_node_id=id, message_type=MessageType.QUERY, group_id="Query") for id in node_ids]
    out = []

    for r in grid.send_and_receive(msgs, timeout=3600):
        if r.has_error():
            raise RuntimeError(f"Un cliente fallo en el query: {r.error}")

        metrics = dict(r.content["metrics"])
        out.append({"node_id": r.metadata.src_node_id, "pid": int(metrics["pid"]), "n": float(metrics["n"]), "m": float(metrics["m"])})

    return sorted(out, key=lambda x: x["pid"])


def make_server_model(model: str, d: int, sets: dict) -> WoeServer | TreeServer:
    srv = WoeServer(d) if model == "woe_logreg" else TreeServer(model, d)
    
    for name, x in sets.tems():
        srv.bind(name, x)

    return srv

def cosine(a:np.ndarray, b:np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na==0 or nb == 0 else float(a @ b / (na * nb))

def gini(x: np.ndarray) -> float:
    x =np.sort(np.asarray(x, dtype=float))
    if x.sum() == 0:
        return 0.0

    k = np.arange(1, len(x) + 1)

    return float((2 * (k*x).sum() / (len(x) * x.sum())) - (len(x)+1) / len(x))


def run_simulation(grid: Grid, config: dict, nodes: list[dict], system:str, strategy: str, xtest: np.ndarray, ytest: np.ndarray, xva: np.ndarray, yva: np.ndarray, central_auc: float, max_rounds:int, budget_frac: float, fixed_share: float) -> tuple[list[dict], dict]:
    K = len(nodes)
    seed, model = int(config["seed"]), config["model"]
    n = np.array([x["n"] for x in nodes])
    prices = np.ones(K) if strategy == "all" else make_prices(n, np.array([x["m"] for x in nodes]), fixed_share)
    B = budget_frac * prices.sum()
    cm = ContributionModel(n)
    srv = make_server_model(model, len(feature_names(config["part-dir"])), {"test": xtest, "va":xva})
    mon = ConvergenceMonitor()
    sel_rng = np.random.default_rng(np.random.SeedSequence([seed, K, 8001, zlib.crc32(f"{system}|{model}".encode())]))

    rows = []
    stop = False
    for t in range(1, max_rounds + 1):
        cm.start_round()
        srv.start_round()
        if strategy == "all":
            selected = list(range(K))
        elif strategy == "random":
            selected = select_random(prices, B, sel_rng)
        elif strategy == "greedy":
            selected = select_greedy(cm.values(t), prices, B)
        else:
            select_lp(cm.vaues(t), prices, B)

        msgs = []
        for k in selected:
            conf = ConfigRecord({
                "round": t,
                "model": model,
                "seed": seed,
                "system": system,
                "part-dir": config["part-dir"],
                "need_woe": srv.need_woe(k),
                "no-drop": int(strategy == "all"),
                "start": srv.start(k),
                "exp-id": f"{system}|{strategy}|{budget_frac}|{fixed_share}",
            })

            msgs.append(Message(content=RecordDict({"arrays": ArrayRecord(srv.payload(k)), "config": conf}), dst_node_id=nodes[k]["node_id"], message_type=MessageType.TRAIN, group_id=str(t)))

        node2k = {x["node_id"]: k for k, x in enumerate(nodes)}

        replies = {}
        for r in (grid.send_and_receive(msgs, timeout=3600) if msgs else []):
            if r.has_error():
                raise RuntimeError(f"Un cliene fallo en train (ronda{t}): {r.error}")
            
            if float(dict(r.content["metrics"])["dropped"]) == 0.0:
                replies[node2k[r.metadata.src_node_id]] = r

        responded = sorted(replies)
        cm.observe_sttempts(selected, responded)

        if responded:
            vecs = srv.absorb(responded, replies, n)
            if len(responded) >= 2:
                tot = sum(n[k] * v for k, v in zip(responded, vecs))
                for k, v in zip(responded, vecs):
                    cm.observe_quality(k, (1.0 + cosine(v, tot - n[k] + v)) / 2.0)

        metrics = score_metrics(ytest, srv.margin("test"))
        metrics_va = score_metrics(yva, srv.margin("va"))

        rows.appennd({"ronda": t,
                    **metrics,
                    "auc_valid": metrics_va["roc_auc"],
                    "loglos_va": metrics_va["logloss"],
                    "gasto": float(prices[selected].sum()) if strategy != "all" else float(len(selected)),
                    "presupuesto": float(B),
                    "n_seleccionados": len(selected),
                    "n_respondieron": len(responded),
                    "seleccionados": "|".join(map(str, selected))})

        if mon.update(t, metrics_va["roc_auc"]):
            stop = True
            break

    counts = np.zeros(K)
    for r in rows:
        for k in filter(None, r["seleccionados"].split("|")):
            counts[int(k)] += 1

    auc = np.array([r["roc_auc"] for r in rows])
    t_star = mon.t_star or len(rows)

    summary = {"rondas_max": int(max_rounds),
               "rondas_ejecutadas": len(rows),
               "rondas_conv": int(t_star),
               "convergio": int(stop),
               "auc_conv": float(auc[t_star - 1]),
               "ratio_conv_vs_central": float(auc[t_star - 1] / central_auc),
               "gasto_hasta_conv": float(sum(r["gasto"] for r in rows[:t_star])),
               "auc_final": float(auc[-1]), "auc_max": float(np.nanmax(auc)),
               "auc_prom_ult5": float(np.nanmean(auc[-5:])),
               "ratio_vs_central": float(auc[-1] / central_auc),
               "gasto_total": float(sum(r["gasto"] for r in rows)),
               "presupuesto_ronda": float(B),
               "prom_respondieron": float(np.mean([r["n_respondieron"] for r in rows])),
               "clientes_alcanzados": int((counts > 0).sum()), 
               "gini_participacion": gini(counts)}

    return rows, summary


def append_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, mode="a", header=not path.exists(), index=False)

@app.main()
def main(grid: Grid, context:Context) -> None:
    config = dict(context.run_config)
    for key in ("part-dir", "out-dir", "model-params"):
        if config.get(key):
            config[key] = abs_path(config[key]).as_posix()

    out_dir = abs_path(config["out-dir"])
    out_dir.mkdir(parents=True, exists_ok=True)

    model, max_rounds, K = config["model"], int(config["rounds"]), int(config["K"])

    part = f"K{int(config['K'])}_{Path(config['part-dir']).name}"

    runs_f, rounds_f = out_dir / "runs.csv", out_dir / "rounds.csv"
    
    done = set()
    if runs_f.exists():
        r = pd.read_csv(runs_f)
        done = set(zip(r["particion"], r["model"], r["sistema"], r["estrategia"], r["presupuesto_frac"].round(4), r["costo_fijo"].round(4)))

    variants = config.get("variants", "base")
    combos = [(b, f) for b in BUDGET_FRACTIONS for f in FIXED_COST_SHARES]
    is_base = lambda c: c == (DEFAULT_BUDGET_FRACTION, DEFAULT_FIXED_COST_SHARE)
    combos = [c for c in combos if variants == "all" or (variants == "base") == is_base(c)]
    
    todo = [("sin_sistema", "all", *REF_VAL)] if variants in ("base", "all") else []
    todo += [(s, st, b, f) for (b, f) in combos for s in SYSTEM_LEVELS for st in STRATEGIES]
    todo = [x for x in todo if (part, model, x[0], x[1], round(x[2], 4), round(x[3], 4)) not in done]
    if not todo:
        print(f"{part} / {model}: Ya esta completo")
        return

    xtest, ytest = load_pooled(config["part-dir"], "global_split", "test") # Esto solo se hace para reportar metricas en cada ronda pero no se toma en cuenta en la simulación.
    xva, yva = load_pooled(config["part-dir"], "local_split", "valid") # Esto es solo para detectar la convergencia del modelo pero no afecta la selección de clientes.
    central_auc = central_baseline(model, config, xtest, ytest, out_dir)

    meta = {
        "particion": part,
        "K": K,
        "heterogeneidad_datos": config["level"],
        "escenario": config["scenario"],
        "semilla": int(config["seed"]),
        "modelo": model,
        }

    nodes_by_system: dict[str, list[dict]] = {}
    
    print(f"{part} (K={K}, datos={config['level']}) | modelo={model} | hasta {max_rounds} rondas | n_k: ", end="")
    for system, strategy, b_frac, f_share in todo:
        qsys = "baja" if system == "sin_sistema" else system

        if qsys not in nodes_by_system:
            nodes_by_system[qsys] = query_nodes(grid, config, qsys)

        nodes = nodes_by_system[qsys]
        if system == "sin_sistema" and not nodes_by_system.get("_PRINTED"):
            print(f"{int(min(x['n'] for x in nodes))} .. {int(max(x['n'] for x in nodes))}")
            nodes_by_system["_PRINTED"] = True

        rows, summ = run_simulation(grid, config, nodes, system, strategy, xtest, ytest, xva, yva, central_auc, max_rounds, b_frac, f_share)

        var = {"presupuesto_frac": b_frac, "costo_fijo": f_share}

        append_csv(rounds_f, pd.DataFrame([{**meta, "sistema": system, "estrategia": strategy, **var, **r} for r in rows]))
        append_csv(runs_f, pd.DataFrame([{**meta, "sistema": system, "estrategia": strategy, **var, **summ}]))

        print(f"""{system:<12} {strategy:<7} b={b_frac:<4} F={f_share:<4} AUC={summ['auc_final']:.4f} ({summ['ratio_vs_central']:.1%} del central)
                      conv={summ['rondas_conv']}/{summ['rondas_ejecutadas']}{'' if summ['convergio'] else ' (MAX)'}
                      gasto/ronda={summ['gasto_total'] / summ['rondas_ejecutadas']:.2f} (B={summ['presupuesto_ronda']:.2f}) """)