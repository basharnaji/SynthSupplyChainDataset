"""
Synthetic Multi-Echelon Supply Chain Planning Dataset Generator
Exact regeneration version (seed = 42)

Run:
    python generate_dataset.py --output_dir ./data --weeks 156 --start_date=2023-01-02 --seed 42
"""

import argparse
import os
import math
import random
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--weeks", type=int, default=156)
    parser.add_argument("--start_date", type=str, default="2023-01-02")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------
    # FIXED CONFIG
    # ------------------
    random.seed(args.seed)
    np.random.seed(args.seed)

    products = ["P1", "P2", "P3", "P4", "P5"]
    regions = ["West", "Central", "East"]
    dcs = ["DC_W", "DC_C", "DC_E"]
    region_to_dc = {"West": "DC_W", "Central": "DC_C", "East": "DC_E"}

    week_dates = pd.date_range(
        pd.to_datetime(args.start_date),
        periods=args.weeks,
        freq="W-MON"
    )

    # ------------------
    # PRODUCT MASTER
    # ------------------
    products_df = pd.DataFrame({
        "product": products,
        "uom": ["unit"] * len(products),
        "base_demand": np.random.randint(180, 520, size=len(products)),
        "season_amp": np.random.uniform(0.08, 0.22, size=len(products)),
        "trend_per_week": np.random.uniform(-0.25, 0.45, size=len(products)),
        "promo_sensitivity": np.random.uniform(0.10, 0.35, size=len(products)),
    })
    products_df.to_csv(f"{args.out_dir}/products.csv", index=False)

    # ------------------
    # COSTS
    # ------------------
    costs_df = pd.DataFrame({
        "product": products,
        "holding_cost_per_unit_week": np.round(
            np.random.uniform(0.10, 0.55, size=len(products)), 3
        ),
        "stockout_cost_per_unit": np.round(
            np.random.uniform(8.0, 22.0, size=len(products)), 2
        ),
    })
    costs_df.to_csv(f"{args.out_dir}/costs.csv", index=False)

    # ------------------
    # PROMOTIONS
    # ------------------
    promo_rows = []
    for w, d in enumerate(week_dates):
        for p in products:
            promo_rows.append({
                "week": w,
                "week_start": d.date().isoformat(),
                "product": p,
                "promo_intensity": int(random.random() < 0.10),
            })
    promotions_df = pd.DataFrame(promo_rows)
    promotions_df.to_csv(f"{args.out_dir}/promotions.csv", index=False)

    # ------------------
    # DEMAND GENERATION
    # ------------------
    demand_rows = []
    region_share = np.array([0.38, 0.30, 0.32])

    for w, d in enumerate(week_dates):
        seasonal = math.sin(2 * math.pi * (w % 52) / 52.0)

        for p in products:
            row = products_df.loc[products_df["product"] == p].iloc[0]

            mean = row["base_demand"] * (1 + row["season_amp"] * seasonal)
            mean += row["trend_per_week"] * w

            promo_flag = promotions_df.loc[
                (promotions_df["week"] == w) &
                (promotions_df["product"] == p),
                "promo_intensity"
            ].iloc[0]

            if promo_flag == 1:
                mean *= (1 + row["promo_sensitivity"])

            total_demand = max(0, np.random.normal(mean, 0.15 * mean))
            shares = np.random.dirichlet(15 * region_share)

            for r, sh in zip(regions, shares):
                demand_rows.append({
                    "week": w,
                    "week_start": d.date().isoformat(),
                    "region": r,
                    "product": p,
                    "demand": int(round(total_demand * sh)),
                })

    demand_df = pd.DataFrame(demand_rows)
    demand_df.to_csv(f"{args.out_dir}/demand.csv", index=False)

    # ------------------
    # INVENTORY STATE (DC LEVEL)
    # ------------------
    state_rows = []
    for w, d in enumerate(week_dates):
        for dc in dcs:
            for p in products:
                state_rows.append({
                    "week": w,
                    "week_start": d.date().isoformat(),
                    "location": dc,
                    "product": p,
                    "on_hand": int(np.random.uniform(200, 600)),
                    "in_transit": int(np.random.uniform(50, 200)),
                    "backlog": int(np.random.uniform(0, 50)),
                })

    inventory_state_df = pd.DataFrame(state_rows)
    inventory_state_df.to_csv(
        f"{args.out_dir}/inventory_state.csv", index=False
    )

    # ------------------
    # ACTIONS (BASE-STOCK-LIKE)
    # ------------------
    action_rows = []
    for w, d in enumerate(week_dates):
        for dc in dcs:
            for p in products:
                action_rows.append({
                    "week": w,
                    "week_start": d.date().isoformat(),
                    "action_type": "replenishment_order",
                    "origin": "F1",
                    "destination": dc,
                    "product": p,
                    "qty": int(np.random.uniform(100, 400)),
                })

    actions_df = pd.DataFrame(action_rows)
    actions_df.to_csv(f"{args.out_dir}/actions.csv", index=False)

    # ------------------
    # REWARDS
    # ------------------
    reward_rows = []
    for w, d in enumerate(week_dates):
        for dc in dcs:
            for p in products:
                hc = costs_df.loc[
                    costs_df["product"] == p,
                    "holding_cost_per_unit_week"
                ].iloc[0]
                sc = costs_df.loc[
                    costs_df["product"] == p,
                    "stockout_cost_per_unit"
                ].iloc[0]

                holding_cost = hc * 200
                stockout_cost = sc * 5

                reward_rows.append({
                    "week": w,
                    "week_start": d.date().isoformat(),
                    "location": dc,
                    "product": p,
                    "holding_cost": round(holding_cost, 2),
                    "stockout_cost": round(stockout_cost, 2),
                    "reward": round(-(holding_cost + stockout_cost), 2),
                })

    rewards_df = pd.DataFrame(reward_rows)
    rewards_df.to_csv(f"{args.out_dir}/rewards.csv", index=False)

    # ------------------
    # README MARKER
    # ------------------
    with open(f"{args.out_dir}/README.txt", "w") as f:
        f.write(
            "Synthetic Multi-Echelon Supply Chain Planning Dataset\n"
            f"Generated with seed={args.seed}\n"
        )

    print("Dataset generation complete.")
    print("Output directory:", args.out_dir)


if __name__ == "__main__":
    main()