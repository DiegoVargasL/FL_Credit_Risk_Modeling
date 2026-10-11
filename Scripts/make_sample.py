from __future__ import annotations

import argparse
import shutil

import pandas as pd

from common_pipeline import ID, RANDOM_SEED, get_paths, setup_console

CHUNK = 500_000

BY_CLIENT = ["previous_application.csv", "installments_payments.csv",
             "POS_CASH_balance.csv", "credit_card_balance.csv"]

def filter_csv(src, dst, column, keep: set) -> set:
    first, kept, ids = True, 0, set()
    for chunk in pd.read_csv(src, chunksize=CHUNK):
        sub = chunk[chunk[column].isin(keep)]
        if "SK_ID_BUREAU" in sub.columns and column == ID:
            ids.update(sub["SK_ID_BUREAU"].tolist())
        sub.to_csv(dst, mode="w" if first else "a", header=first, index=False)
        first, kept =  False, kept + len(sub)
    print(f"{src.name}: {kept:,} filas conservadas")
    return ids

def main() -> None:
    setup_console()

    parser = argparse.ArgumentParser(description="Crea data/sample a partir de data/raw")
    parser.add_argument("--n-train",  type=int, default=10_000, help="Número de aplicaciones en aplication_tain.")
    parser.add_argument("--n-test",  type=int, default=2_000, help="Número de aplicaciones en aplication_test.")
    args = parser.parse_args()

    src_dir, dst_dir = get_paths(False).raw, get_paths(True).raw
    dst_dir.mkdir(parents=True, exist_ok=True)

    ids = set()
    for name, n in (("application_train.csv",  args.n_train),  ("application_test.csv", args.n_test)):
        df = pd.read_csv(src_dir / name)
        df = df.sample(min(n, len(df)),  random_state=RANDOM_SEED)
        df.to_csv(dst_dir / name, index=False)
        ids.update(df[ID].tolist())
        print(f"{name}: {len(df):,} clientes.")

    for name in BY_CLIENT:
        filter_csv(src_dir / name, dst_dir / name, ID, ids)

    bureau_ids = filter_csv(src_dir / "bureau.csv", dst_dir / "bureau.csv", ID, ids)
    filter_csv(src_dir / "bureau_balance.csv", dst_dir / "bureau_balance.csv", "SK_ID_BUREAU", bureau_ids)

    desc =  src_dir / "HomeCredit_columns_descriptions.csv"
    if desc.exists():
        shutil.copy(desc, dst_dir / desc.name)
    print(f"Listo. El nuevo subconjunto esta en {dst_dir}")


if __name__ == "__main__":
    main()