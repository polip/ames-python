# ames-python

Ames Housing price prediction: Random Forest / HistGradientBoosting model served via FastAPI on Cloud Run.

- **Train**: `python train_model_v1.py [--no-upload]` (benchmarks RF vs HGB, saves the best)
- **Serve**: `uvicorn ames_fastapi:app` or `python ames_fastapi.py`
- **Test**: `pytest`
- **Lint/format**: `ruff check . && ruff format .`
- **Typecheck**: `mypy ames_fastapi.py train_model_v1.py`
- **Deploy**: `python deploy_api_gcs.py --deploy`

## Training pipeline (`train_model_v1.py`)

- Data prep: drops the 4 documented `GrLivArea > 4000` outliers (`--keep-outliers` to disable)
  and applies domain-aware fills (`NA` = "no such feature" in 15 categorical and 9 numeric columns).
- Target: `SalePrice` in dollars, modeled as `log` internally via `TransformedTargetRegressor`
  (numpy `np.log`/`np.exp`, so the pickle unloads cleanly anywhere) — the served model predicts
  dollars directly.
- A/B benchmark in one MLflow run (`fast 5-fold shuffled CV`, single `cross_validate` call):
  Random Forest (`n_estimators=300, max_features=0.35`) vs `HistGradientBoostingRegressor`;
  the winner (lowest dollar RMSLE) is fitted on the full data.
- Metrics (current run, HGB selected): R² = **0.902**, RMSLE = **0.128**, MAPE = **9.1 %** (RF: 0.887 / 0.133 / 9.3 %).
- Artifacts:
  - `model/ames-rf-model.pkl` — fitted pipeline, joblib-compressed (~0.8 MB for HGB).
  - `model/ames-rf-model.meta.json` — serving contract: feature list, feature defaults
    (median/mode), `target_transform`, MLflow `run_id`. Consumed by the API.
  - MLflow: params per candidate + metrics `rf_*` / `hgb_*`, feature schema, model with signature.
- CLI: `--data`, `--out-dir`, `--model {auto,rf,hgb}`, `--n-estimators`, `--max-features`,
  `--min-samples-leaf`, `--jobs` (CV-level parallelism; inner estimators stay single-threaded),
  `--keep-outliers`, `--no-upload` (run with real GCS upload to push versioned blobs to
  `gs://scikit-models/models/…` plus the stable `ames-rf-model.pkl` pointer).

## Serving contract (`ames_fastapi.py`)

- Loads the model and the meta JSON from GCS (local `model/` fallback for both).
  The service account's `roles/storage.objectViewer` binding lets GCS loading succeed;
  the image-baked `/app/model/` copy is the cold-fail-safe.
- Input rows are completed with training-time `feature_defaults` for fields the client omits,
  then reindexed to the schema's exact feature order.
- `target_transform` in meta decides dollar conversion: `identity` (current HGB/RF pipeline)
  or legacy `log10` (older artifacts, e.g. the previously deployed forest) are both supported.
- Models without `estimators_`/`feature_importances_` (i.e. HGB, or any Pipeline) return
  `confidence_interval: null` / `feature_importance: null` instead of erroring.

## Tests

- `tests/test_api.py` — API endpoints (dummy forest injected, legacy behavior).
- `tests/test_pipeline.py` — training smoke tests: outlier rule, domain fills, RF/HGB fit +
  dollar predictions, CV metric keys, joblib round-trip, serving-contract (meta) validity.

## Deployment notes

- `requirements_gcloud.txt` **pins** `scikit-learn==1.7.2`, `joblib==1.5.2`, `numpy==2.3.3`
  to the training venv — sklearn pickles are not cross-version compatible (unpinning breaks
  container startup with `No module named '_loss'`).
- The Cloud Run service account has `roles/storage.objectViewer` on `gs://scikit-models`
  so it loads model + meta from GCS instead of the image-baked fallback.
- Known pre-existing quirk: `deploy-api-gcs.py --deploy` raises `FileNotFoundError` at the
  very end (`'gcloud logs tail'` invoked as one string) — after the service is already
  deployed, safe to ignore.

See [AGENTS.md](AGENTS.md) for full details.


