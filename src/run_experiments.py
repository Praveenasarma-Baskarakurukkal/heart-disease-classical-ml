from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "heart+disease" / "processed.cleveland.data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_COLUMN = "num"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
FEATURE_SELECTION_K = 8
SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add compact, clinically motivated interaction features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=FEATURE_COLUMNS)

        X = X.copy()
        safe_age = X["age"].replace(0, np.nan)
        safe_trestbps = X["trestbps"].replace(0, np.nan)

        X["age_oldpeak"] = X["age"] * X["oldpeak"]
        X["thalach_age_ratio"] = X["thalach"] / safe_age
        X["chol_trestbps_ratio"] = X["chol"] / safe_trestbps
        X["risk_cp"] = (X["cp"] == 4).astype(int)
        X["exercise_risk"] = X["exang"] * X["oldpeak"]
        X["vessel_thal_combo"] = X["ca"].fillna(0) * X["thal"].fillna(0)
        X["high_oldpeak"] = (X["oldpeak"] >= 2.0).astype(int)

        return X


@dataclass
class EvalResult:
    model_name: str
    variant: str
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def ensure_dirs():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH, header=None, names=ALL_COLUMNS, na_values=["?"])
    y = (df[TARGET_COLUMN] > 0).astype(int)
    X = df[FEATURE_COLUMNS].copy()
    return X, y


def baseline_preprocessor(feature_selection: str = "none") -> ColumnTransformer:
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]

    if feature_selection == "none":
        numeric_steps.append(("scaler", StandardScaler()))
    elif feature_selection == "chi2":
        numeric_steps.append(("scaler", MinMaxScaler()))
    else:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_COLUMNS),
            ("cat", Pipeline(categorical_steps), CATEGORICAL_COLUMNS),
        ]
    )


def improved_preprocessor() -> ColumnTransformer:
    engineered_numeric = NUMERIC_COLUMNS + [
        "age_oldpeak",
        "thalach_age_ratio",
        "chol_trestbps_ratio",
        "exercise_risk",
        "vessel_thal_combo",
    ]
    engineered_categorical = CATEGORICAL_COLUMNS + ["risk_cp", "high_oldpeak"]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                engineered_numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                engineered_categorical,
            ),
        ]
    )


def make_baseline_pipeline(model, feature_selection: str = "none") -> Pipeline:
    steps: list[tuple[str, object]] = [
        ("preprocess", baseline_preprocessor(feature_selection)),
    ]

    if feature_selection == "chi2":
        steps.append(("select", SelectKBest(score_func=chi2, k=FEATURE_SELECTION_K)))
    elif feature_selection == "mutual_info":
        steps.append(("select", SelectKBest(score_func=mutual_info_classif, k=FEATURE_SELECTION_K)))

    steps.append(("model", model))
    return Pipeline(steps)


def make_improved_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("feature_engineering", ClinicalFeatureEngineer()),
            ("preprocess", improved_preprocessor()),
            ("model", model),
        ]
    )


def evaluate_predictions(model_name: str, variant: str, y_true, y_pred, y_score) -> EvalResult:
    return EvalResult(
        model_name=model_name,
        variant=variant,
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_score),
    )


def bootstrap_metric_intervals(y_true, y_pred, y_score, n_bootstrap: int = 2000, seed: int = RANDOM_STATE) -> dict:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    n = len(y_true)
    metrics = {"accuracy": [], "balanced_accuracy": [], "f1": [], "roc_auc": []}

    attempts = 0
    while len(metrics["accuracy"]) < n_bootstrap and attempts < n_bootstrap * 5:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        metrics["accuracy"].append(accuracy_score(yt, yp))
        metrics["balanced_accuracy"].append(balanced_accuracy_score(yt, yp))
        metrics["f1"].append(f1_score(yt, yp, zero_division=0))
        metrics["roc_auc"].append(roc_auc_score(yt, ys))

    summary = {}
    for metric_name, values in metrics.items():
        values = np.asarray(values)
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "ci_lower": float(np.percentile(values, 2.5)),
            "ci_upper": float(np.percentile(values, 97.5)),
        }
    return summary


def cross_validate_pipeline(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold) -> dict:
    cv_result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)
    return {
        "cv_mean_accuracy": float(np.mean(cv_result["test_accuracy"])),
        "cv_std_accuracy": float(np.std(cv_result["test_accuracy"])),
        "cv_mean_balanced_accuracy": float(np.mean(cv_result["test_balanced_accuracy"])),
        "cv_std_balanced_accuracy": float(np.std(cv_result["test_balanced_accuracy"])),
        "cv_mean_precision": float(np.mean(cv_result["test_precision"])),
        "cv_std_precision": float(np.std(cv_result["test_precision"])),
        "cv_mean_recall": float(np.mean(cv_result["test_recall"])),
        "cv_std_recall": float(np.std(cv_result["test_recall"])),
        "cv_mean_f1": float(np.mean(cv_result["test_f1"])),
        "cv_std_f1": float(np.std(cv_result["test_f1"])),
        "cv_mean_roc_auc": float(np.mean(cv_result["test_roc_auc"])),
        "cv_std_roc_auc": float(np.std(cv_result["test_roc_auc"])),
    }


def metrics_to_frame(results: list[EvalResult]) -> pd.DataFrame:
    return pd.DataFrame([vars(r) for r in results]).sort_values(
        ["f1", "roc_auc", "accuracy"], ascending=[False, False, False]
    )


def plot_class_balance(y: pd.Series):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    counts = y.value_counts().sort_index()
    ax = sns.barplot(x=["No disease", "Disease"], y=counts.values, hue=["No disease", "Disease"], palette=["#4C78A8", "#F58518"], legend=False)
    ax.set_title("Binary Class Distribution")
    ax.set_ylabel("Patients")
    for idx, value in enumerate(counts.values):
        ax.text(idx, value + 2, str(value), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_balance.png", dpi=300)
    plt.close()


def plot_missingness(X: pd.DataFrame):
    missing = X.isna().sum().sort_values(ascending=False)
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(x=missing.index, y=missing.values, color="#72B7B2")
    ax.set_title("Missing Values in Processed Cleveland Data")
    ax.set_ylabel("Missing entries")
    ax.tick_params(axis="x", rotation=45)
    for idx, value in enumerate(missing.values):
        ax.text(idx, value + 0.05, str(int(value)), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "missingness.png", dpi=300)
    plt.close()


def plot_correlation_heatmap(X: pd.DataFrame, y: pd.Series):
    combined = X.copy()
    combined["target"] = y
    corr = combined.apply(pd.to_numeric, errors="coerce").corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()


def baseline_experiments(X_train, X_test, y_train, y_test) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    baseline_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=RANDOM_STATE
        ),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }
    variants = ["none", "chi2", "mutual_info"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results: list[EvalResult] = []
    cv_rows: list[dict] = []
    detailed_outputs: dict[str, dict] = {}

    for variant in variants:
        for name, estimator in baseline_models.items():
            pipeline = make_baseline_pipeline(clone(estimator), feature_selection=variant)
            cv_summary = cross_validate_pipeline(pipeline, X_train, y_train, cv)
            cv_rows.append({"model_name": name, "variant": variant, **cv_summary})
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_score = pipeline.predict_proba(X_test)[:, 1]
            result = evaluate_predictions(name, variant, y_test, y_pred, y_score)
            results.append(result)

            key = f"{variant}::{name}"
            detailed_outputs[key] = {
                "metrics": vars(result),
                "cv_metrics": {"model_name": name, "variant": variant, **cv_summary},
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
                "bootstrap_intervals": bootstrap_metric_intervals(y_test, y_pred, y_score),
                "y_score": y_score.tolist(),
                "y_pred": y_pred.tolist(),
            }

    frame = metrics_to_frame(results)
    cv_frame = pd.DataFrame(cv_rows).sort_values(
        ["cv_mean_f1", "cv_mean_roc_auc", "cv_mean_accuracy"], ascending=[False, False, False]
    )
    frame.to_csv(TABLES_DIR / "baseline_holdout_results.csv", index=False)
    cv_frame.to_csv(TABLES_DIR / "baseline_cv_results.csv", index=False)
    with open(RESULTS_DIR / "baseline_detailed_results.json", "w", encoding="utf-8") as handle:
        json.dump(detailed_outputs, handle, indent=2)
    return frame, cv_frame, detailed_outputs


def make_feature_engineering_lr_pipeline(class_weight=None, C: float = 1.0, penalty: str = "l2") -> Pipeline:
    return Pipeline(
        [
            ("feature_engineering", ClinicalFeatureEngineer()),
            ("preprocess", improved_preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=1500,
                    solver="liblinear",
                    class_weight=class_weight,
                    C=C,
                    penalty=penalty,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def ablation_experiments(
    tuned_models: dict,
    ensemble: VotingClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    ablations = {
        "Baseline Logistic Regression": make_baseline_pipeline(
            LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
            feature_selection="none",
        ),
        "Feature Engineering Only": make_feature_engineering_lr_pipeline(class_weight=None, C=1.0, penalty="l2"),
        "Class Weights Only": make_baseline_pipeline(
            LogisticRegression(
                max_iter=1500,
                solver="liblinear",
                class_weight="balanced",
                C=0.5,
                penalty="l1",
                random_state=RANDOM_STATE,
            ),
            feature_selection="none",
        ),
        "Feature Eng. + Class Weights": clone(tuned_models["Balanced Logistic Regression"]),
        "Soft Voting Ensemble": ensemble,
    }

    rows = []
    for name, pipeline in ablations.items():
        cv_summary = cross_validate_pipeline(pipeline, X_train, y_train, cv)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_score = pipeline.predict_proba(X_test)[:, 1]
        holdout = evaluate_predictions(name, "ablation", y_test, y_pred, y_score)
        rows.append({**vars(holdout), **cv_summary})

    ablation_df = pd.DataFrame(rows).sort_values(["cv_mean_f1", "f1", "roc_auc"], ascending=[False, False, False])
    ablation_df.to_csv(TABLES_DIR / "ablation_results.csv", index=False)
    return ablation_df


def improved_experiments(X_train, X_test, y_train, y_test) -> tuple[pd.DataFrame, dict, dict, VotingClassifier]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    candidate_grids = {
        "Balanced Logistic Regression": (
            LogisticRegression(max_iter=1500, solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE),
            {
                "model__C": [0.1, 0.5, 1.0, 2.0, 5.0],
                "model__penalty": ["l1", "l2"],
            },
        ),
        "Balanced SVM": (
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            {
                "model__C": [0.5, 1.0, 2.0],
                "model__gamma": ["scale", 0.05, 0.1],
            },
        ),
        "Balanced Random Forest": (
            RandomForestClassifier(class_weight="balanced_subsample", random_state=RANDOM_STATE),
            {
                "model__n_estimators": [200, 300],
                "model__max_depth": [None, 4, 6],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt"],
            },
        ),
    }

    tuning_rows = []
    tuned_models = {}

    for name, (estimator, params) in candidate_grids.items():
        pipeline = make_improved_pipeline(clone(estimator))
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=params,
            scoring={"f1": "f1", "roc_auc": "roc_auc", "balanced_accuracy": "balanced_accuracy"},
            refit="f1",
            cv=cv,
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        tuned_models[name] = search.best_estimator_
        tuning_rows.append(
            {
                "model_name": name,
                "best_params": json.dumps(search.best_params_),
                "cv_mean_f1": search.best_score_,
                "cv_std_f1": search.cv_results_["std_test_f1"][search.best_index_],
                "cv_mean_balanced_accuracy": search.cv_results_["mean_test_balanced_accuracy"][search.best_index_],
                "cv_std_balanced_accuracy": search.cv_results_["std_test_balanced_accuracy"][search.best_index_],
                "cv_mean_roc_auc": search.cv_results_["mean_test_roc_auc"][search.best_index_],
                "cv_std_roc_auc": search.cv_results_["std_test_roc_auc"][search.best_index_],
            }
        )

    tuning_df = pd.DataFrame(tuning_rows).sort_values("cv_mean_f1", ascending=False)
    tuning_df.to_csv(TABLES_DIR / "improved_model_cv_results.csv", index=False)

    top_two_names = tuning_df.head(2)["model_name"].tolist()
    top_two = []
    for name in top_two_names:
        base_estimator = tuned_models[name]
        calibrated = CalibratedClassifierCV(estimator=base_estimator, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE))
        top_two.append((name, calibrated))

    ensemble = VotingClassifier(estimators=top_two, voting="soft")
    ensemble.fit(X_train, y_train)

    holdout_rows = []
    roc_curves = []
    fitted_models = {"Soft Voting Ensemble": ensemble}

    for name in top_two_names:
        estimator = clone(tuned_models[name])
        estimator.fit(X_train, y_train)
        fitted_models[name] = estimator

    for name, estimator in fitted_models.items():
        y_pred = estimator.predict(X_test)
        y_score = estimator.predict_proba(X_test)[:, 1]
        holdout_rows.append(vars(evaluate_predictions(name, "improved", y_test, y_pred, y_score)))
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_curves.append((name, fpr, tpr))

    holdout_df = pd.DataFrame(holdout_rows).sort_values(["f1", "roc_auc"], ascending=False)
    holdout_df.to_csv(TABLES_DIR / "improved_holdout_results.csv", index=False)
    best_model_name = holdout_df.iloc[0]["model_name"]
    best_model = fitted_models[best_model_name]

    plot_roc_curves(roc_curves)
    export_feature_rankings(tuned_models[top_two_names[0]], X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_score = best_model.predict_proba(X_test)[:, 1]
    summary = {
        "best_model_name": best_model_name,
        "top_two_cv_models": top_two_names,
        "holdout_best_metrics": holdout_df.iloc[0].to_dict(),
        "best_model_confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "best_model_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "best_model_bootstrap_intervals": bootstrap_metric_intervals(y_test, y_pred, y_score),
    }
    with open(RESULTS_DIR / "improved_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return holdout_df, summary, tuned_models, ensemble


def plot_roc_curves(curves):
    plt.figure(figsize=(6, 5))
    for name, fpr, tpr in curves:
        plt.plot(fpr, tpr, linewidth=2, label=name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves on the Holdout Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=300)
    plt.close()


def export_feature_rankings(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series):
    model.fit(X_train, y_train)
    preprocess = model.named_steps["preprocess"]
    feature_names = preprocess.get_feature_names_out()
    estimator = model.named_steps["model"]

    if hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_[0])
    elif hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    else:
        return

    ranking = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    ranking.to_csv(TABLES_DIR / "top_features.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=ranking, x="importance", y="feature", color="#54A24B")
    plt.title("Top Engineered Features")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_features.png", dpi=300)
    plt.close()


def write_dataset_summary(X: pd.DataFrame, y: pd.Series):
    summary = {
        "instances": int(len(X)),
        "features": len(X.columns),
        "positive_rate": float(y.mean()),
        "missing_values": X.isna().sum().to_dict(),
        "class_counts": y.value_counts().sort_index().to_dict(),
    }
    with open(RESULTS_DIR / "dataset_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    numeric_summary = X.describe(include="all").transpose()
    numeric_summary.to_csv(TABLES_DIR / "dataset_descriptives.csv")


def main():
    ensure_dirs()
    X, y = load_data()
    write_dataset_summary(X, y)
    plot_class_balance(y)
    plot_missingness(X)
    plot_correlation_heatmap(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    baseline_results, baseline_cv_results, baseline_detail = baseline_experiments(X_train, X_test, y_train, y_test)
    improved_results, improved_summary, tuned_models, ensemble = improved_experiments(X_train, X_test, y_train, y_test)
    ablation_results = ablation_experiments(tuned_models, ensemble, X_train, X_test, y_train, y_test)

    benchmark_rf = baseline_results[
        (baseline_results["model_name"] == "Random Forest") & (baseline_results["variant"] == "none")
    ].iloc[0].to_dict()
    baseline_cv_best = baseline_cv_results.iloc[0].to_dict()

    overall_summary = {
        "dataset_source": str(DATA_PATH),
        "baseline_best": baseline_results.iloc[0].to_dict(),
        "baseline_cv_best": baseline_cv_best,
        "benchmark_comparable_random_forest": benchmark_rf,
        "improved_best": improved_results.iloc[0].to_dict(),
        "best_ablation_cv": ablation_results.iloc[0].to_dict(),
        "notes": [
            "processed.cleveland.data is used because the provided WARNING file marks cleveland.data as corrupted.",
            "Binary classification follows the original UCI convention: num > 0 indicates presence of heart disease.",
            "Feature-selection settings in the benchmark paper are under-specified, so the reproduction uses a standard SelectKBest top-8 setup.",
            "Cross-validation for baseline characterization and hyperparameter tuning is performed on the training partition only; the holdout test partition remains untouched until final evaluation.",
        ],
        "available_detail_files": [
            "results/baseline_detailed_results.json",
            "results/improved_summary.json",
            "results/dataset_summary.json",
        ],
    }
    with open(RESULTS_DIR / "overall_summary.json", "w", encoding="utf-8") as handle:
        json.dump(overall_summary, handle, indent=2)

    print("Saved results to:", RESULTS_DIR)
    print("Best baseline:", overall_summary["baseline_best"])
    print("Best improved:", improved_summary["holdout_best_metrics"])


if __name__ == "__main__":
    main()
