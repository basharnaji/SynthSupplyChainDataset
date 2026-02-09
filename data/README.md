# Synthetic Multi-Echelon Supply Chain Planning Dataset

This dataset simulates a **weekly** supply chain with the following stages:

**Supplier → Factory → Distribution Center (DC) → Customer Regions**

It is designed to support:
- Demand forecasting
- Inventory/service analytics (fill rate, backlog, costs)
- Baseline planning policy evaluation
- Offline RL / sequential decision making research

## Key settings
- Weeks: 156
- Start date: 2025-01-02 (Mondays)
- Seed: 42
- Products: 5 (Product1, Product2, Product3, Product4, Product5)
- Regions: 3 (West, Central, East)
- Factories: 2 (Factory1, Factory2)
- DCs: 3 (DC_W, DC_C, DC_E)

## Mapping
Regions are served by a single DC:
- West → DC_W
- Central → DC_C
- East → DC_E

## Files (what each one is for)

### 1) `demand.csv`
Customer demand by week/region/product.

### 2) `promotions.csv`
Promo schedule used by the demand generator. `promo_intensity` ∈ {0,1,2}.

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
