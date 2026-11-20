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
inventory snapshot and product master tables came with it.

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
