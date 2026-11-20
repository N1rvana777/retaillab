# RetailLab

Demand forecasting and replenishment decisions for a regional supermarket chain:
quantile forecasts of daily store-SKU demand feed a newsvendor order rule.

> **Public snapshot.** This is a sanitized copy of a private research repository.
> The data came from a partner chain that has since withdrawn permission to use it, so
> the data, trained models, reports, research logs and partner-facing documents are not
> included, and store names, locations, brand names and business figures have been
> removed. Stores and cities appear as neutral codes (`store_a`–`store_c`, `city_a`,
> `city_b`). The pipeline cannot run end to end without the original data; the tests use
> synthetic data.

## Overview

- **Decision.** How much each store should order of each product each day. A stock-out loses margin and overstock costs spoilage and holding, so the order is the demand quantile at the critical fractile `Cu/(Cu+Co)`, estimated directly by quantile regression.
- **Data.** Three stores, January 2024 to February 2026: about 15 million store-SKU-day rows built from 292 POS export files, plus purchase records, inventory snapshots and product master data from an ERP with its own product codes.
- **Censored demand.** Sales understate demand on stock-out days. The v2 pipeline reconstructs daily inventory from purchases, sales and snapshots, and treats a day as stocked out when the opening stock is zero.
- **Ordering.** A newsvendor optimiser with minimum-order, shelf and truck constraints, unit-tested against closed-form solutions.
- **Audit.** An audit in September 2026 withdrew the headline profit numbers. Stock-out labels used statistics pooled over both years and a forward-looking window; some inputs, such as the day's discount and observed weather, are known only after the day ends; and the test year had been used to prune features and choose the quantile.

The project is closed. The partner withdrew permission to use the data, so the planned fixes were not carried out.

## Contents

- `data_pipeline/`: ingestion of POS (Kemai) exports and ERP (Guanjiapo) tables; cleaning
  (misrecords, outages, discount anomalies); feature engineering with a leakage guard
  (lags shifted by one day within store-SKU groups, lunar-calendar holiday alignment,
  weather); and the v2 pipeline in `ingest_v2/` (daily inventory reconstruction, observed
  stock-outs, SKU master and categories).
- `quant_engine/`: LightGBM / XGBoost quantile forecasting, cold start, ensemble;
  newsvendor optimisation at the critical fractile `Cu/(Cu+Co)`; constraints; safety-stock
  fallback; drift monitoring; decision telemetry.
- `api_gateway/`: FastAPI service with JWT authentication and rate limiting.
- `scripts/`: training, backtests, censored-demand estimation (`demand_uncensor.py`),
  profit evaluation (`profit_v2.py`), quantile comparisons, sales-pattern analysis.
- `tests/`: unit and integration tests on synthetic data.
- `notebooks/`: a simulated blind test on mock data.
- `ARCHITECTURE.md` (Chinese): pipeline and module design. `CLAUDE.md` and `AGENTS.md`
  are the rules given to the AI coding agents; some files they mention are not public.

## Data (not included)

Daily SKU-level sales from three stores for 2024 and 2025: about 14 million transaction
rows, aggregated to about 6.8 million (2024) and 7.1 million (2025) store-SKU-day rows.
About 43,000 products were sold in both years, and 64,000 to 71,000 in each year. Purchase,
inventory snapshot and product master tables came with it. The feature table, which also
covers January and February 2026, has about 15.2 million rows.

## Results

Trained on 2024 and evaluated on 2025, the LightGBM quantile models had a mean absolute
error of 4.52 units per store-SKU-day. The 10%, 50% and 90% quantile forecasts covered
15.2%, 52.8% and 90.2% of outcomes (`backtests/training_log.csv`).

An audit in September 2026 withdrew the earlier simulated profit improvements (+8.7%, and
about +3.9% for v2). They were scenario numbers built on estimated demand and a chain-wide
cost ratio, not realised gains.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests -q
```

Much of the code was written by AI coding agents (Claude Code, Codex) working under the
rules in `CLAUDE.md` and `AGENTS.md`.

## License

MIT; see `LICENSE`.
