"""Measure the scorer against many held-out MovieLens users, not one export.

    python -m scripts.eval_synthetic
    python -m scripts.eval_synthetic --users 100 --repeats 1     # quick
    python -m scripts.eval_synthetic --compare                   # old vs new

WHY THIS SCRIPT EXISTS

One Letterboxd export puts n=11 in the Q3 cell. Every knob in this project was
being chosen by comparing numbers computed from eleven films, which is not
measurement, it is decoration. A few hundred held-out MovieLens users turn that
into thousands of observations and make the central question -- does
personalisation beat recommending acclaimed films to everybody -- actually
answerable.

The users scored here were excluded from the similarity build by
scripts.build_cf, so this is not testing the system on its own training data.
The harness refuses to run if that is not true.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from lbxd import movielens
from lbxd.cf import ItemItemCF
from lbxd.config import CF, DATA_PROCESSED
from lbxd.synthetic import (
    HOLDOUT_USERS_NPY,
    evaluate_users,
    pick_holdout_users,
    verify_disjoint,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--compare", action="store_true",
                    help="run the old scorer and the new one side by side")
    ap.add_argument("--beta", type=float, default=None)
    args = ap.parse_args()

    ml = movielens.load()
    path = DATA_PROCESSED / HOLDOUT_USERS_NPY
    if not path.exists():
        raise SystemExit(
            f"{path} missing -- run `python -m scripts.build_cf` first so that "
            "the evaluation users are excluded from the similarity matrix."
        )
    excluded = np.load(path)
    users = pick_holdout_users(ml)[: args.users]
    verify_disjoint(ml, users, excluded)

    cf = ItemItemCF.load()
    print(f"evaluating {len(users)} held-out users x {args.repeats} splits\n")

    configs = {"current": CF}
    if args.compare:
        configs = {
            "LEGACY (z input, no shrinkage)": replace(
                CF, use_residual_input=False, consensus_beta=0.0, pred_shrinkage_k=0.0
            ),
            "residual, beta=1.0": replace(CF, use_residual_input=True, consensus_beta=1.0),
            "residual, beta=0.5": replace(CF, use_residual_input=True, consensus_beta=0.5),
            "residual, beta=0.0": replace(CF, use_residual_input=True, consensus_beta=0.0),
        }
    if args.beta is not None:
        configs = {f"beta={args.beta}": replace(CF, consensus_beta=args.beta)}

    for name, cfg in configs.items():
        cf.config = cfg
        res = evaluate_users(ml, cf, users, n_repeats=args.repeats)
        print("=" * 78)
        print(name)
        print("=" * 78)
        print(res.summary())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
