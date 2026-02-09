"""
Synthetic Multi-Echelon Supply Chain Planning Dataset Generator (Full)

Generates a 4-stage network:
Supplier -> Factory -> DC -> Customer Regions

Outputs:
- demand, promotions
- capacity constraints
- lead times (base)
- routing/sourcing
- policy targets + initial inventory
- simulated weekly operations producing:
  - actions (replenishment orders + production)
  - inventory_state (on_hand / in_transit / backlog / arrivals)
  - rewards (holding + stockout cost, reward = -total cost)
  - shipments_planned (using base lead time)

Run:
  python generate_dataset.py --out_dir ./data --weeks 156 --start_date 2023-01-02 --seed 42
"""

import argparse
import json
import math
import os
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def draw_lead_time(base: int) -> int:
    """
    Stochastic lead-time variation around the base lead time.
    - Usually base or base-1
    - Occasional +1
    - Rare disruption spike +2..+5
    """
    lt = base + np.random.choice([-1, 0, 0, 0, 1], p=[0.10, 0.55, 0.20, 0.10, 0.05])
    lt = max(0, int(lt))
    if random.random() < 0.02:
        lt += random.randint(2, 5)
    return lt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--weeks", type=int, default=156)
    ap.add_argument("--start_date", type=str, default="2025-01-02")  # Monday
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Determinism
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ----------------------------
    # Network configuration
    # ----------------------------
    products = ["Product1", "Product2", "Product3", "Product4", "Product5"]
    regions = ["West", "Central", "East"]
    suppliers = ["Supplier1", "Supplier2"]
    factories = ["Factory1", "Factory2"]
    dcs = ["DC_W", "DC_C", "DC_E"]
    region_to_dc = {"West": "DC_W", "Central": "DC_C", "East": "DC_E"}

    week_dates = pd.date_range(pd.to_datetime(args.start_date), periods=args.weeks, freq="W-MON")

    # ----------------------------
    # products.csv
    # ----------------------------
    products_df = pd.DataFrame(
        {
            "product": products,
            "uom": ["unit"] * len(products),
            "base_demand": np.random.randint(180, 520, size=len(products)),
            "season_amp": np.random.uniform(0.08, 0.22, size=len(products)),
            "trend_per_week": np.random.uniform(-0.25, 0.45, size=len(products)),
            "promo_sensitivity": np.random.uniform(0.10, 0.35, size=len(products)),
        }
    )
    products_df.to_csv(os.path.join(args.out_dir, "products.csv"), index=False)

    # ----------------------------
    # costs.csv
    # ----------------------------
    costs_df = pd.DataFrame(
        {
            "product": products,
            "holding_cost_per_unit_week": np.round(np.random.uniform(0.10, 0.55, size=len(products)), 3),
            "stockout_cost_per_unit": np.round(np.random.uniform(8.0, 22.0, size=len(products)), 2),
            "production_cost_per_unit": np.round(np.random.uniform(2.0, 7.0, size=len(products)), 2),
            "transport_cost_per_unit": np.round(np.random.uniform(0.3, 1.2, size=len(products)), 2),
        }
    )
    costs_df.to_csv(os.path.join(args.out_dir, "costs.csv"), index=False)

    # ----------------------------
    # sourcing.csv and routing.csv
    # ----------------------------
    sourcing_df = pd.DataFrame(
        [
            {
                "product": p,
                "primary_supplier": (pref := random.choice(suppliers)),
                "secondary_supplier": [s for s in suppliers if s != pref][0],
            }
            for p in products
        ]
    )
    routing_df = pd.DataFrame(
        [
            {
                "product": p,
                "primary_factory": (pref := random.choice(factories)),
                "secondary_factory": [f for f in factories if f != pref][0],
            }
            for p in products
        ]
    )
    sourcing_df.to_csv(os.path.join(args.out_dir, "sourcing.csv"), index=False)
    routing_df.to_csv(os.path.join(args.out_dir, "routing.csv"), index=False)

    prod_primary_factory = dict(zip(routing_df["product"], routing_df["primary_factory"]))

    # ----------------------------
    # lead_times.csv (BASE lead times)
    # ----------------------------
    lt_rows = []
    # Supplier -> Factory
    for s in suppliers:
        for f in factories:
            for p in products:
                lt_rows.append(
                    {
                        "origin": s,
                        "destination": f,
                        "product": p,
                        "base_lead_time_weeks": random.randint(2, 5),
                    }
                )
    # Factory -> DC
    for f in factories:
        for dc in dcs:
            for p in products:
                lt_rows.append(
                    {
                        "origin": f,
                        "destination": dc,
                        "product": p,
                        "base_lead_time_weeks": random.randint(1, 3),
                    }
                )
    # DC -> Region (0 if the region is served by that DC else 1)
    for dc in dcs:
        for r in regions:
            for p in products:
                base = 0 if region_to_dc[r] == dc else 1
                lt_rows.append(
                    {
                        "origin": dc,
                        "destination": r,
                        "product": p,
                        "base_lead_time_weeks": int(base),
                    }
                )

    lead_times_df = pd.DataFrame(lt_rows)
    lead_times_df.to_csv(os.path.join(args.out_dir, "lead_times.csv"), index=False)

    base_lt: Dict[Tuple[str, str, str], int] = {
        (row.origin, row.destination, row.product): int(row.base_lead_time_weeks)
        for row in lead_times_df.itertuples(index=False)
    }

    # ----------------------------
    # promotions.csv
    # promo_intensity: 0/1/2
    # ----------------------------
    promo_rows = []
    for w, d in enumerate(week_dates):
        for p in products:
            val = 0
            if random.random() < 0.10:
                val = 1 if random.random() < 0.75 else 2
            promo_rows.append({"week": w, "week_start": d.date().isoformat(), "product": p, "promo_intensity": val})
    promotions_df = pd.DataFrame(promo_rows)
    promotions_df.to_csv(os.path.join(args.out_dir, "promotions.csv"), index=False)

    # ----------------------------
    # demand.csv
    # Demand has:
    # - seasonality
    # - promotions
    # - regime shifts
    # - rare spikes
    # ----------------------------
    region_share = np.array([0.38, 0.30, 0.32])
    region_share = region_share / region_share.sum()

    cp1 = random.randint(35, 60)
    cp2 = random.randint(95, 120)
    shock1 = np.random.uniform(-0.12, 0.18, size=len(products))
    shock2 = np.random.uniform(-0.10, 0.22, size=len(products))

    demand_rows = []
    for w, d in enumerate(week_dates):
        week_of_year = int(d.isocalendar().week)
        seasonal = math.sin(2 * math.pi * week_of_year / 52.0)

        for pi, p in enumerate(products):
            row = products_df.loc[products_df["product"] == p].iloc[0]
            base = float(row["base_demand"])
            amp = float(row["season_amp"])
            trend = float(row["trend_per_week"])
            promo_sens = float(row["promo_sensitivity"])

            promo_int = int(
                promotions_df.loc[(promotions_df["week"] == w) & (promotions_df["product"] == p), "promo_intensity"].iloc[
                    0
                ]
            )

            regime = 1.0
            if w >= cp1:
                regime *= (1.0 + float(shock1[pi]))
            if w >= cp2:
                regime *= (1.0 + float(shock2[pi]))

            spike = 1.0
            if random.random() < 0.02:
                spike *= random.uniform(1.25, 1.70)

            mean = (base * regime) * (1.0 + amp * seasonal) + trend * w + (promo_sens * promo_int * base * 0.10)
            mean = max(0.0, mean)

            noise = np.random.normal(0, 0.12 * mean + 12)
            total = max(0.0, mean + noise) * spike

            shares = np.random.dirichlet(15 * region_share)
            for r, sh in zip(regions, shares):
                demand_rows.append(
                    {
                        "week": w,
                        "week_start": d.date().isoformat(),
                        "region": r,
                        "product": p,
                        "demand": int(round(total * sh)),
                    }
                )

    demand_df = pd.DataFrame(demand_rows)
    demand_df.to_csv(os.path.join(args.out_dir, "demand.csv"), index=False)

    # ----------------------------
    # capacity.csv
    # Factory capacity varies with:
    # - planned shutdown
    # - random outage
    # - rare yield loss
    # Then is allocated across products via a Dirichlet split.
    # ----------------------------
    cap_rows = []
    for w, d in enumerate(week_dates):
        for f in factories:
            base_total = 2400 if f == "F1" else 2100

            planned = 0.25 if (w % 52) in [25, 26] else 0.0
            outage = random.uniform(0.10, 0.45) if random.random() < 0.04 else 0.0
            yield_loss = random.uniform(0.05, 0.18) if random.random() < 0.02 else 0.0

            total = base_total * (1.0 - planned) * (1.0 - outage) * (1.0 - yield_loss)
            weights = np.random.dirichlet(np.ones(len(products)) * 3.5)

            for p, wt in zip(products, weights):
                cap_rows.append(
                    {
                        "week": w,
                        "week_start": d.date().isoformat(),
                        "site": f,
                        "product": p,
                        "capacity": int(round(total * wt)),
                    }
                )

    capacity_df = pd.DataFrame(cap_rows)
    capacity_df.to_csv(os.path.join(args.out_dir, "capacity.csv"), index=False)

    # ----------------------------
    # policy_targets.csv
    # base_stock_target per DC/product, computed from:
    #   target ≈ mean_demand * (avg_lt + safety_weeks)
    # ----------------------------
    mean_dem = demand_df.groupby(["region", "product"])["demand"].mean().reset_index()
    mean_dem["dc"] = mean_dem["region"].map(region_to_dc)
    dc_mean = (
        mean_dem.groupby(["dc", "product"])["demand"]
        .sum()
        .reset_index()
        .rename(columns={"demand": "mean_weekly_demand"})
    )

    f2dc = lead_times_df[lead_times_df["origin"].isin(factories) & lead_times_df["destination"].isin(dcs)]
    f2dc_mean = (
        f2dc.groupby(["destination", "product"])["base_lead_time_weeks"]
        .mean()
        .reset_index()
        .rename(columns={"destination": "dc", "base_lead_time_weeks": "lt"})
    )

    targets = dc_mean.merge(f2dc_mean, on=["dc", "product"], how="left")
    targets["safety_weeks"] = np.random.uniform(1.0, 2.5, size=len(targets))
    targets["base_stock_target"] = (targets["mean_weekly_demand"] * (targets["lt"] + targets["safety_weeks"])).round().astype(int)

    policy_targets_df = targets[["dc", "product", "mean_weekly_demand", "lt", "safety_weeks", "base_stock_target"]]
    policy_targets_df.to_csv(os.path.join(args.out_dir, "policy_targets.csv"), index=False)

    # ----------------------------
    # initial_inventory.csv
    # ----------------------------
    locations = factories + dcs
    init_rows = []
    for loc in locations:
        for p in products:
            if loc in dcs:
                tgt = int(policy_targets_df.loc[(policy_targets_df["dc"] == loc) & (policy_targets_df["product"] == p), "base_stock_target"].iloc[0])
                onh = int(round(np.random.uniform(0.5, 1.0) * tgt))
            else:
                onh = int(round(np.random.uniform(300, 900)))
            init_rows.append({"location": loc, "product": p, "initial_on_hand": onh})
    initial_inventory_df = pd.DataFrame(init_rows)
    initial_inventory_df.to_csv(os.path.join(args.out_dir, "initial_inventory.csv"), index=False)

    # ----------------------------
    # Simulation producing:
    # - actions.csv
    # - inventory_state.csv
    # - rewards.csv
    # plus shipments_planned.csv (base LT view)
    # ----------------------------
    on_hand = {(loc, p): int(initial_inventory_df.loc[(initial_inventory_df["location"] == loc) & (initial_inventory_df["product"] == p), "initial_on_hand"].iloc[0])
               for loc in locations for p in products}
    backlog = {(dc, p): 0 for dc in dcs for p in products}
    pipelines = []  # list of dicts {origin,destination,product,ship_week,arrival_week,qty}

    actions_rows = []
    state_rows = []
    reward_rows = []

    max_order_mult = 1.8

    for w, d in enumerate(week_dates):
        # arrivals
        new_pipes = []
        arrivals = {(loc, p): 0 for loc in locations for p in products}
        for rec in pipelines:
            if rec["arrival_week"] == w:
                dest = rec["destination"]
                if dest in locations:
                    on_hand[(dest, rec["product"])] += rec["qty"]
                    arrivals[(dest, rec["product"])] += rec["qty"]
            else:
                new_pipes.append(rec)
        pipelines = new_pipes

        # demand at DCs (aggregate regions -> DC)
        wk_dem = demand_df[demand_df["week"] == w].copy()
        wk_dem["dc"] = wk_dem["region"].map(region_to_dc)
        dc_dem = wk_dem.groupby(["dc", "product"])["demand"].sum().to_dict()

        shipments = {(dc, p): 0 for dc in dcs for p in products}
        unmet = {(dc, p): 0 for dc in dcs for p in products}

        # fulfill at DCs
        for dc in dcs:
            for p in products:
                dem = int(dc_dem.get((dc, p), 0))
                need = dem + backlog[(dc, p)]
                serve = min(on_hand[(dc, p)], need)
                on_hand[(dc, p)] -= serve
                shipments[(dc, p)] = serve
                back = need - serve
                backlog[(dc, p)] = back
                unmet[(dc, p)] = back

        # compute on-order to DC (pipeline qty destined to DC)
        on_order = {(dc, p): 0 for dc in dcs for p in products}
        for rec in pipelines:
            if rec["destination"] in dcs:
                on_order[(rec["destination"], rec["product"])] += rec["qty"]

        # DC replenishment orders (Factory -> DC)
        for dc in dcs:
            for p in products:
                tgt = int(policy_targets_df.loc[(policy_targets_df["dc"] == dc) & (policy_targets_df["product"] == p), "base_stock_target"].iloc[0])
                ip = on_hand[(dc, p)] + on_order[(dc, p)] - backlog[(dc, p)]  # inventory position
                desired = max(0, tgt - ip)
                cap = int(round(max_order_mult * tgt))
                order_qty = int(min(desired, cap))

                if order_qty > 0:
                    f = prod_primary_factory[p]
                    lt = draw_lead_time(base_lt[(f, dc, p)])
                    pipelines.append({"origin": f, "destination": dc, "product": p, "ship_week": w, "arrival_week": w + lt, "qty": order_qty})

                actions_rows.append({
                    "week": w,
                    "week_start": d.date().isoformat(),
                    "action_type": "replenishment_order",
                    "origin": prod_primary_factory[p],
                    "destination": dc,
                    "product": p,
                    "qty": order_qty,
                })

        # factory production to meet same-week outbound shipments
        ship_reqs = {(f, p): 0 for f in factories for p in products}
        for rec in pipelines:
            if rec["origin"] in factories and rec["ship_week"] == w:
                ship_reqs[(rec["origin"], rec["product"])] += rec["qty"]

        for f in factories:
            for p in products:
                req = ship_reqs[(f, p)]
                avail = on_hand[(f, p)]
                need = max(0, req - avail)
                cap = int(capacity_df.loc[(capacity_df["week"] == w) & (capacity_df["site"] == f) & (capacity_df["product"] == p), "capacity"].iloc[0])
                prod_qty = int(min(need, cap))

                on_hand[(f, p)] += prod_qty

                ship_qty = min(on_hand[(f, p)], req)
                on_hand[(f, p)] -= ship_qty

                # if we couldn't ship everything, reduce pipeline qty for those shipments
                short = req - ship_qty
                if short > 0:
                    rem = short
                    for rec in pipelines:
                        if rem <= 0:
                            break
                        if rec["origin"] == f and rec["product"] == p and rec["ship_week"] == w:
                            take = min(rec["qty"], rem)
                            rec["qty"] -= take
                            rem -= take
                    pipelines = [rec for rec in pipelines if rec["qty"] > 0]

                actions_rows.append({
                    "week": w,
                    "week_start": d.date().isoformat(),
                    "action_type": "production",
                    "origin": f,
                    "destination": f,
                    "product": p,
                    "qty": prod_qty,
                })

        # in_transit by destination
        in_transit = {(loc, p): 0 for loc in locations for p in products}
        for rec in pipelines:
            if rec["destination"] in locations:
                in_transit[(rec["destination"], rec["product"])] += rec["qty"]

        # log state
        for loc in locations:
            for p in products:
                state_rows.append({
                    "week": w,
                    "week_start": d.date().isoformat(),
                    "location": loc,
                    "product": p,
                    "on_hand": int(on_hand[(loc, p)]),
                    "in_transit": int(in_transit[(loc, p)]),
                    "backlog": int(backlog[(loc, p)]) if loc in dcs else 0,
                    "arrivals": int(arrivals[(loc, p)]),
                })

        # reward at DCs
        for dc in dcs:
            for p in products:
                hc = float(costs_df.loc[costs_df["product"] == p, "holding_cost_per_unit_week"].iloc[0])
                sc = float(costs_df.loc[costs_df["product"] == p, "stockout_cost_per_unit"].iloc[0])

                holding_cost = hc * on_hand[(dc, p)]
                stockout_cost = sc * unmet[(dc, p)]
                reward_rows.append({
                    "week": w,
                    "week_start": d.date().isoformat(),
                    "location": dc,
                    "product": p,
                    "holding_cost": round(holding_cost, 3),
                    "stockout_cost": round(stockout_cost, 3),
                    "reward": round(-(holding_cost + stockout_cost), 3),
                    "served_units": int(shipments[(dc, p)]),
                    "unmet_units": int(unmet[(dc, p)]),
                })

    actions_df = pd.DataFrame(actions_rows)
    inventory_state_df = pd.DataFrame(state_rows)
    rewards_df = pd.DataFrame(reward_rows)

    actions_df.to_csv(os.path.join(args.out_dir, "actions.csv"), index=False)
    inventory_state_df.to_csv(os.path.join(args.out_dir, "inventory_state.csv"), index=False)
    rewards_df.to_csv(os.path.join(args.out_dir, "rewards.csv"), index=False)

    # shipments_planned.csv: use BASE LT for the replenishment order rows (not stochastic)
    rep = actions_df[actions_df["action_type"] == "replenishment_order"].copy()
    rep["base_lead_time_weeks"] = rep.apply(lambda r: base_lt[(r["origin"], r["destination"], r["product"])], axis=1)
    rep["ship_week"] = rep["week"]
    rep["arrival_week"] = rep["week"] + rep["base_lead_time_weeks"]
    rep[["ship_week", "arrival_week", "origin", "destination", "product", "qty", "base_lead_time_weeks", "week_start"]] \
        .rename(columns={"week_start": "ship_week_start"}) \
        .to_csv(os.path.join(args.out_dir, "shipments_planned.csv"), index=False)

    # ----------------------------
    # data_dictionary.json (machine readable)
    # ----------------------------
    data_dictionary = {
        "products.csv": {
            "product": "Product identifier (Product1..Product5).",
            "uom": "Unit of measure.",
            "base_demand": "Baseline weekly demand level used by demand generator.",
            "season_amp": "Seasonality amplitude (fraction of base demand).",
            "trend_per_week": "Linear trend applied per week.",
            "promo_sensitivity": "Multiplicative lift factor applied during promos.",
        },
        "costs.csv": {
            "product": "Product identifier.",
            "holding_cost_per_unit_week": "Holding cost per unit per week at DCs.",
            "stockout_cost_per_unit": "Penalty per unit of unmet demand/backlog at DCs.",
            "production_cost_per_unit": "Unit production cost (not used in reward by default).",
            "transport_cost_per_unit": "Unit transport cost (not used in reward by default).",
        },
        "sourcing.csv": {
            "product": "Product identifier.",
            "primary_supplier": "Primary supplier assigned to the product.",
            "secondary_supplier": "Secondary supplier assigned to the product.",
        },
        "routing.csv": {
            "product": "Product identifier.",
            "primary_factory": "Primary factory assigned to the product.",
            "secondary_factory": "Secondary factory assigned to the product.",
        },
        "lead_times.csv": {
            "origin": "Origin node (supplier/factory/DC).",
            "destination": "Destination node (factory/DC/region).",
            "product": "Product identifier.",
            "base_lead_time_weeks": "Base lead time in weeks (used in shipments_planned; sim uses stochastic variation around this).",
        },
        "promotions.csv": {
            "week": "0-based week index.",
            "week_start": "ISO date for week start (Monday).",
            "product": "Product identifier.",
            "promo_intensity": "0=no promo, 1=promo, 2=strong promo.",
        },
        "demand.csv": {
            "week": "0-based week index.",
            "week_start": "ISO date for week start (Monday).",
            "region": "Customer region (West/Central/East).",
            "product": "Product identifier.",
            "demand": "Realized demand units in that week/region/product.",
        },
        "capacity.csv": {
            "week": "0-based week index.",
            "week_start": "ISO date for week start (Monday).",
            "site": "Factory site (F1/F2).",
            "product": "Product identifier.",
            "capacity": "Effective production capacity (units/week) available for that product at that site.",
        },
        "policy_targets.csv": {
            "dc": "Distribution center (DC_W/DC_C/DC_E).",
            "product": "Product identifier.",
            "mean_weekly_demand": "Average weekly demand into the DC (aggregated across its regions).",
            "lt": "Average base lead time (factory->DC) used in target calculation.",
            "safety_weeks": "Extra weeks of safety stock coverage.",
            "base_stock_target": "Target inventory position for base-stock policy.",
        },
        "initial_inventory.csv": {
            "location": "Factory or DC.",
            "product": "Product identifier.",
            "initial_on_hand": "Starting on-hand inventory before week 0 simulation.",
        },
        "actions.csv": {
            "week": "0-based week index.",
            "week_start": "ISO date for week start (Monday).",
            "action_type": "replenishment_order (factory->DC) or production (within factory).",
            "origin": "Origin node (factory).",
            "destination": "Destination node (DC for orders; factory for production).",
            "product": "Product identifier.",
            "qty": "Action quantity (units).",
        },
        "inventory_state.csv": {
            "week": "0-based week index.",
            "week_start": "ISO date for week start (Monday).",
            "location": "Factory or DC.",
            "product": "Product identifier.",
            "on_hand": "End-of-week on-hand inventory at location.",
            "in_transit": "Pipeline inventory destined to this location (all open shipments).",
            "backlog": "Unmet demand backlog (DC only).",
            "arrivals": "Units received/arrived at this location in the current week.",
        },
        "rewards.csv": {
            "week": "0-based week index.",
            "week_start": "ISO date for week start (Monday).",
            "location": "DC (DC_W/DC_C/DC_E).",
            "product": "Product identifier.",
            "holding_cost": "Holding cost incurred at the DC this week.",
            "stockout_cost": "Stockout/backlog penalty this week.",
            "reward": "Reward = -(holding_cost + stockout_cost).",
            "served_units": "Units actually shipped/served to customers from the DC this week.",
            "unmet_units": "Units unmet this week (added to backlog).",
        },
        "shipments_planned.csv": {
            "ship_week": "Week index when the order was placed/shipped (same as decision week).",
            "arrival_week": "Planned arrival week using BASE lead time.",
            "origin": "Factory origin.",
            "destination": "DC destination.",
            "product": "Product identifier.",
            "qty": "Order quantity.",
            "base_lead_time_weeks": "Base lead time used to compute arrival_week.",
            "ship_week_start": "ISO date for ship_week start (Monday).",
        },
    }

    with open(os.path.join(args.out_dir, "data_dictionary.json"), "w") as f:
        json.dump(data_dictionary, f, indent=2)

    # ----------------------------
    # README.md (human readable)
    # ----------------------------
    readme = f"""# Synthetic Multi-Echelon Supply Chain Planning Dataset

This dataset simulates a **weekly** supply chain with the following stages:

**Supplier → Factory → Distribution Center (DC) → Customer Regions**

It is designed to support:
- Demand forecasting
- Inventory/service analytics (fill rate, backlog, costs)
- Baseline planning policy evaluation
- Offline RL / sequential decision making research

## Key settings
- Weeks: {args.weeks}
- Start date: {args.start_date} (Mondays)
- Seed: {args.seed}
- Products: {len(products)} ({', '.join(products)})
- Regions: {len(regions)} ({', '.join(regions)})
- Factories: {len(factories)} ({', '.join(factories)})
- DCs: {len(dcs)} ({', '.join(dcs)})

## Mapping
Regions are served by a single DC:
- West → DC_W
- Central → DC_C
- East → DC_E

## Files (what each one is for)

### 1) `demand.csv`
Customer demand by week/region/product.

### 2) `promotions.csv`
Promo schedule used by the demand generator. `promo_intensity` ∈ {{0,1,2}}.

### 3) `capacity.csv`
Factory capacity by week/site/product. Includes planned shutdowns + random outages/yield loss.

### 4) `lead_times.csv`
Base lead times (weeks) between nodes. Simulation uses stochastic variation around base.

### 5) `routing.csv`, `sourcing.csv`
Reference mappings assigning each product to primary/secondary factory and supplier.

### 6) `policy_targets.csv`
Base-stock targets per DC/product derived from mean demand + (lead time + safety coverage).

### 7) `initial_inventory.csv`
Starting inventory by location/product.

### 8) `actions.csv`
Behavior policy decisions:
- `replenishment_order`: factory → DC order quantities
- `production`: factory production quantities (capacity constrained)

### 9) `inventory_state.csv`
Weekly state snapshots (useful for forecasting features or RL state):
- On-hand
- In-transit pipeline
- Backlog at DCs
- Arrivals

### 10) `rewards.csv`
Costs and reward at each DC/product/week:
- holding_cost, stockout_cost
- reward = -(holding_cost + stockout_cost)

### 11) `shipments_planned.csv`
A “planned arrivals” view of replenishment orders using **base** lead time.

## Column definitions
See `data_dictionary.json` for a complete machine-readable schema.

## RL framing (optional)
You can view each DC-product as an MDP with weekly steps:
- State: (on_hand, in_transit, backlog, demand history, promo flags, etc.)
- Action: replenishment order qty
- Reward: negative cost from `rewards.csv`
"""
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write(readme)

    print("Done. Wrote dataset to:", args.out_dir)


if __name__ == "__main__":
    main()