from __future__ import annotations

import zlib

import numpy as np
from flwr.app import Message, Context, RecordDict, MetricRecord, ArrayRecord, ConfigRecord
from flwr.clientapp import ClientApp

from task import (load_client_train, infra_factor, drop_prob, drop_draw, local_counts, woe_features,
                  sgd_logreg_update, train_block, ensemble_margin)


app = ClientApp()


def _reply(msg:Message, metrics: dict, **records) -> Message:
    return Message(content=RecordDict({"metrics": MetricRecord(metrics), **records}), reply_to=msg)


@app.query()
def query(msg: Message, context:Context) -> Message:
    config = dict(msg.content["config"])
    pid = int(context.node_config["partition-id"])
    x, y = load_client_train(config["part-dir"], pid)

    return _reply(msg, {"n": float(len(y)), "pid": float(pid), "m": infra_factor(config["system"], int(config["seed"]), pid)})

@app.train()
def train(msg: Message, context: Context) -> Message:
    config = dict(msg.content["config"])
    pid = int(context.node_config["partition-id"])
    seed, t, model = int(config["seed"]), int(config["round"]), config["model"]

    # Determinar el drop-out del cliente
    if not int(config["no-drop"]) and drop_draw(seed, pid, t) < drop_prob(config["system"], seed, pid):
        return _reply(msg, {"dropped": 1.0, "pid": float(pid), "n": 0.0})

    x, y = load_client_train(config["part-dir"], pid)
    arrays = msg.content["arrays"].to_numpy_ndarrays()
    metrics = {"dropped": 0.0, "pid": float(pid), "n": float(len(y))}

    if model == "woe_logreg":
        coef = arrays[0]
        tab = {"sel": arrays[1], "c2b": arrays[2], "woe": arrays[3]}
        rng = np.random.default_rng(np.random.SeedSequence([seed, 9001, t, pid]))
        dw, db = sgd_logreg_update(woe_features(x, tab), y, coef[:-1], float(coef[-1]), rng)
        records = {"arrays": ArrayRecord([np.append(dw, db).astype(np.float32)])}

        if int(config["need-woe"]):
            records["counts"] = ArrayRecord([local_counts(x, y)])

        return _reply(msg, metrics, **records)


    weights, blobs = arrays[0], [a.tobytes() for a in arrays[1:]]
    start, exp = int(config["start"]), config["exp-id"]
    if start > 0:
        sync = dict(context.satte["sync"]) if "sync" in context.state else {}
        base = context.state["margin"].to_numpy_ndarrays()[0].copy()

    else:
        base = np.zeros(len(y), dtype=np.float64)

    base += ensemble_margin(model, blobs, weights, x)
    context.state["margin"] = ArrayRecord([base])
    context.state["sync"] = ConfigRecord({"exp-id": exp, "nb": start + len(blobs)})
    tree_seed = zlib.crc32(f"{seed}|{t}|{pid}".encode()) % (2**31 -1)
    blob = train_block(model, x, y, base, tree_seed)

    return _reply(msg, metrics, arrays=ArrayRecord([np.frombuffer(blob, dtype=np.uint8)]))