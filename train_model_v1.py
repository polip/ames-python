"""Ames housing price training pipeline.

Trains a Random Forest (and benchmarks a HistGradientBoosting candidate) on
log10-transformed target data via ``TransformedTargetRegressor`` so the served
model predicts dollars directly. Cross-validation is a single
``cross_validate`` call over all metrics, and every candidate/model/metric is
tracked in MLflow.

Usage:
    python train_model_v1.py                     # train + save + upload to GCS
    python train_model_v1.py --no-upload         # train + save locally only
    python train_model_v1.py --model rf          # skip the HGB A/B benchmark
"""

import argparse
import json
import logging
import math
from pathlib import Path

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_BUCKET = 'scikit-models'
DEFAULT_MODEL_BLOB = 'ames-rf-model.pkl'
DEFAULT_META_BLOB = 'ames-rf-model.meta.json'
CV_FOLDS = 5
CV_SEED = 42
OUTLIER_GR_LIV_AREA = 4000  # documented outliers: partial sales of very large houses

# Ames data description: "NA" cat. values meaning "no such feature", not "missing".
CATEGORICAL_NONE_FILL = [
    'Alley',
    'BsmtQual',
    'BsmtCond',
    'BsmtExposure',
    'BsmtFinType1',
    'BsmtFinType2',
    'FireplaceQu',
    'GarageType',
    'GarageFinish',
    'GarageQual',
    'GarageCond',
    'PoolQC',
    'Fence',
    'MiscFeature',
    'MasVnrType',
]
NUMERIC_NONE_FILL = [
    'MasVnrArea',
    'BsmtFinSF1',
    'BsmtFinSF2',
    'BsmtUnfSF',
    'TotalBsmtSF',
    'BsmtFullBath',
    'BsmtHalfBath',
    'GarageCars',
    'GarageArea',
]

log = logging.getLogger(__name__)


def build_feature_lists(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split feature names into numeric and categorical."""
    numeric = X.select_dtypes(include=[np.number, 'category']).columns.tolist()
    categorical = X.select_dtypes(include=['object', 'string']).columns.tolist()
    return numeric, categorical


def prepare_features(
    df: pd.DataFrame, keep_outliers: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the feature matrix (X) and dollar target (y) from raw training data."""
    n_raw = len(
        df
    )  # Drop the 4 documented GrLivArea > 4000 outliers (they are not prices to model).
    if not keep_outliers:
        df = df[df['GrLivArea'] <= OUTLIER_GR_LIV_AREA].copy()

    # Domain-aware constant fills (not learned -> no CV leakage).
    for col in NUMERIC_NONE_FILL:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    for col in CATEGORICAL_NONE_FILL:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    X = df.drop(columns=['Id', 'SalePrice'], errors='ignore')
    y = df['SalePrice'].astype(float)
    log.info(
        'Data: %d rows (%d outliers dropped), %d features', len(df), n_raw - len(df), X.shape[1]
    )
    return X, y


def create_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """ColumnTransformer for the Random Forest: median impute + one-hot encode."""
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ],
        remainder='drop',
    )


def build_rf_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    n_estimators: int = 300,
    max_features: float | str = 0.35,
    min_samples_leaf: int = 1,
) -> Pipeline:
    """Random Forest on log10(SalePrice); predicts dollars after inverse transform."""
    forest = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        random_state=CV_SEED,
        n_jobs=1,
    )
    regressor = TransformedTargetRegressor(regressor=forest, func=np.log, inverse_func=np.exp)
    return Pipeline(
        steps=[
            ('preprocessor', create_preprocessor(numeric_features, categorical_features)),
            ('regressor', regressor),
        ]
    )


def build_hgb_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    n_estimators: int = 600,
) -> Pipeline:
    """HistGradientBoosting benchmark: native NaN on numerics, ordinal categoricals."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            (
                'cat',
                Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                        (
                            'encoder',
                            OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder='drop',
    )
    regressor = HistGradientBoostingRegressor(
        max_iter=n_estimators,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        random_state=CV_SEED,
    )
    return Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            (
                'regressor',
                TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp),
            ),
        ]
    )


def evaluate(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, n_jobs: int = -1
) -> dict[str, float]:
    """Single cross_validate run computing all metrics at once (RMSLE on dollars)."""
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    result = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=['r2', 'neg_mean_squared_log_error', 'neg_mean_absolute_percentage_error'],
        n_jobs=n_jobs,
    )
    rmsle = math.sqrt(-np.mean(result['test_neg_mean_squared_log_error']))
    return {
        'r2': float(np.mean(result['test_r2'])),
        'rmsle_dollars': float(rmsle),
        'rmse_log10': float(rmsle / math.log(10)),
        'mape': float(-np.mean(result['test_neg_mean_absolute_percentage_error'])),
        'fit_time_s': float(np.mean(result['fit_time'])),
    }


def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame | None:
    """Importance table of the fitted forest. `None` for candidates without it (HGB)."""
    regressor = pipeline.named_steps['regressor']
    base = regressor.regressor_ if hasattr(regressor, 'regressor_') else regressor
    if not hasattr(base, 'feature_importances_'):
        return None
    names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = base.feature_importances_
    assert len(names) == len(importances), (len(names), len(importances))
    return pd.DataFrame({'feature': names, 'importance': importances}).sort_values(
        'importance', ascending=False, ignore_index=True
    )


def plot_feature_importance(importance: pd.DataFrame, path: Path, top_n: int = 15) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    head = importance.head(top_n).iloc[::-1]
    ax.barh(head['feature'], head['importance'])
    ax.set_title(f'Top {top_n} Feature Importances')
    ax.set_xlabel('Importance')
    for i, (name, value) in enumerate(zip(head['feature'], head['importance'], strict=False)):
        ax.text(value, i, f' {value:.4f}', va='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    log.info('Feature importance plot saved to %s', path)


def build_model_meta(
    X: pd.DataFrame,
    selected_model: str,
    metrics: dict[str, float],
    run_id: str,
) -> dict:
    """Serving-side contract description, consumed by `ames_fastapi.py`."""
    numeric, categorical = build_feature_lists(X)
    defaults: dict[str, str | float] = {col: float(X[col].median()) for col in numeric}
    defaults.update({col: str(X[col].mode().iloc[0]) for col in categorical})
    return {
        'model_class': selected_model,
        'target_transform': 'identity',  # pipeline itself inverts log10 -> dollars
        'features': X.columns.tolist(),
        'numeric_features': numeric,
        'categorical_features': categorical,
        'feature_defaults': defaults,
        'metrics': {k: round(v, 6) for k, v in metrics.items()},
        'run_id': run_id,
    }


def upload_to_gcs(local_dir: Path, run_id: str, bucket_name: str = DEFAULT_BUCKET) -> None:
    """Upload model + meta blobs: one versioned copy, one stable pointer."""
    from google.cloud import storage  # lazy import keeps tests/mypy lean

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    model_path = Path(local_dir, DEFAULT_MODEL_BLOB)
    meta_path = Path(local_dir, DEFAULT_META_BLOB)
    for remote, local in (
        (DEFAULT_MODEL_BLOB, model_path),
        (DEFAULT_META_BLOB, meta_path),
        (f'models/ames-rf-model-{run_id}.pkl', model_path),
        (f'models/ames-rf-model-{run_id}.meta.json', meta_path),
    ):
        blob = bucket.blob(remote)
        blob.upload_from_filename(str(local))
    log.info('Model + meta uploaded to gs://%s/ (versioned under models/)', bucket_name)


def model_train(args: argparse.Namespace) -> tuple[Pipeline, dict[str, str]]:
    """Load data, A/B benchmark candidates, fit the winner, log + save artifacts."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    train_df = pd.read_csv(args.data)
    log.info('Loaded training data: %s', train_df.shape)
    X, y = prepare_features(train_df, keep_outliers=args.keep_outliers)
    numeric_features, categorical_features = build_feature_lists(X)
    log.info(
        'Numeric features: %d, categorical features: %d',
        len(numeric_features),
        len(categorical_features),
    )

    # `--max-features` arrives as a CLI string; sklearn expects float or 'sqrt'/'log2'.
    max_features: float | str = (
        float(args.max_features)
        if args.max_features.replace('.', '', 1).isdigit()
        else args.max_features
    )

    candidates: dict[str, Pipeline] = {}
    if args.model in ('rf', 'auto'):
        candidates['rf'] = build_rf_pipeline(
            numeric_features,
            categorical_features,
            n_estimators=args.n_estimators,
            max_features=max_features,
            min_samples_leaf=args.min_samples_leaf,
        )
    if args.model in ('hgb', 'auto'):
        candidates['hgb'] = build_hgb_pipeline(
            numeric_features, categorical_features, n_estimators=600
        )
    log.info('Candidates: %s', list(candidates))

    experiment = mlflow.set_experiment('ames-housing-prices')

    all_metrics: dict[str, dict[str, float]] = {}
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        run_id = run.info.run_id
        mlflow.log_params(
            {
                'n_samples': len(X),
                'n_features': X.shape[1],
                'n_numeric': len(numeric_features),
                'n_categorical': len(categorical_features),
                'outliers_removed': not args.keep_outliers,
                'candidates': ' '.join(candidates),
                'data': str(args.data),
                'cv_folds': CV_FOLDS,
                'cv_seed': CV_SEED,
                'target': 'SalePrice (dollars, log10 internally)',
                'rf_n_estimators': args.n_estimators,
                'rf_max_features': args.max_features,
                'rf_min_samples_leaf': args.min_samples_leaf,
            }
        )

        for name, candidate in candidates.items():
            metrics = evaluate(candidate, X, y, n_jobs=args.jobs)
            all_metrics[name] = metrics
            for key, value in metrics.items():
                mlflow.log_metric(f'{name}_{key}', value)
            log.info('%s: %s', name, json.dumps(metrics))

        # Winner = lowest RMSLE (dollars). A single candidate is just auto-selected itself.
        best = min(all_metrics, key=lambda k: all_metrics[k]['rmsle_dollars'])
        mlflow.log_param('selected_model', best)
        mlflow.log_metric('best_rmsle_dollars', all_metrics[best]['rmsle_dollars'])
        log.info('Selected candidate: %s', best)

        log.info('Fitting selected model on the full training set...')
        winner = candidates[best]
        winner.fit(X, y)

        importance = extract_feature_importance(winner)
        if importance is not None:
            log.info('Top features:\n%s', importance.head(10).to_string(index=False))
            artifacts = Path('artifacts')
            artifacts.mkdir(exist_ok=True)
            fig_path = artifacts / 'feature_importance.png'
            plot_feature_importance(importance, fig_path)
            mlflow.log_artifact(str(fig_path))
            mlflow.log_dict(
                {'importance': importance.to_dict(orient='records')}, 'experiment/importance.json'
            )

        # Serving contract written next to the model.
        meta = build_model_meta(X, best, all_metrics[best], run_id)
        mlflow.log_dict(meta, 'experiment/feature_schema.json')

        mlflow.sklearn.log_model(
            sk_model=winner,
            name='model',
            input_example=X.head(5),
            signature=mlflow.models.infer_signature(X.head(5), winner.predict(X.head(5))),
        )
        mlflow.log_dict({'candidate_metrics': all_metrics}, 'experiment/metrics.json')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / DEFAULT_MODEL_BLOB
    meta_path = out_dir / DEFAULT_META_BLOB
    joblib.dump(winner, model_path, compress=3)
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info('Model saved locally: %s (%.1f MB)', model_path, model_path.stat().st_size / 1e6)
    log.info('Meta saved locally: %s', meta_path)

    if not args.no_upload:
        upload_to_gcs(out_dir, run_id)
    else:
        log.info('Skipping GCS upload (--no-upload)')

    return winner, {'selected_model': best, 'run_id': run_id}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Ames housing price training pipeline')
    parser.add_argument('--data', default='input/train.csv')
    parser.add_argument('--out-dir', default='model')
    parser.add_argument('--model', choices=['auto', 'rf', 'hgb'], default='auto')
    parser.add_argument('--n-estimators', type=int, default=300)
    parser.add_argument('--max-features', default='0.35')
    parser.add_argument('--min-samples-leaf', type=int, default=1)
    parser.add_argument(
        '--jobs', type=int, default=-1, help='cv parallel jobs; inner forest uses 1'
    )
    parser.add_argument('--keep-outliers', action='store_true', help='do not drop GrLivArea>4000')
    parser.add_argument('--no-upload', action='store_true', help='skip GCS upload')
    return parser.parse_args()


if __name__ == '__main__':
    model_pipeline = model_train(parse_args())[0]
