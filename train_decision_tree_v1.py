"""Decision tree v1: predict final_outcome from company snapshot features.

The modeling table is the yearly snapshot panel in data.zip (training_panel.csv).
Each row is one company on 30 June of a year from 2021 to 2025. The label is
that company's eventual final_outcome, shared by every snapshot of the company.

SIC positions, the SIC count, and jurisdiction come from the Companies House
profile cached in data.zip. Primary SIC on that profile matches sic_1 on every
company in the panel. Accounts type and the numeric filing features come from
the snapshot itself, so they are known at the as-of date.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parent
DATA_ZIP = ROOT / "data.zip"
OUT_PATH = ROOT / "outputs" / "v1_results.json"
RANDOM_STATE = 42
TEST_SIZE = 0.30

LABELS = [
    "active",
    "insolvency",
    "compulsory_strike_off",
    "voluntary_strike_off",
]

NUMERIC = [
    "age_years",
    "accounts_filings",
    "avg_days_late_accounts",
    "max_days_late_accounts",
    "days_since_last_accounts",
    "directors_active",
    "director_resignations_12m",
    "director_appointments_12m",
    "address_changes",
    "strike_off_warnings",
    "charges_created",
    "charges_outstanding",
    "filings_total",
    "n_sic_codes",
]

CATEGORICAL = [
    "sic_1",
    "sic_2",
    "sic_3",
    "sic_4",
    "jurisdiction",
    "last_accounts_type",
]

# Shown in the tree text so a split is readable without a SIC manual.
SIC_NAME = {
    "46170": "food-and-drink-agents",
    "46900": "non-specialised-wholesale",
    "47110": "non-specialised-food-retail",
    "50100": "sea-passenger-transport",
    "50300": "inland-passenger-transport",
    "55100": "hotels",
    "56101": "licensed-restaurants",
    "56102": "unlicensed-restaurants-cafes",
    "56103": "take-away-food",
    "56210": "event-catering",
    "56290": "other-food-service",
    "56301": "licensed-clubs",
    "56302": "public-houses-and-bars",
    "70229": "management-consultancy",
    "73110": "advertising-agencies",
    "85510": "sports-and-recreation-education",
    "95110": "computer-repair",
    "95210": "consumer-electronics-repair",
    "96040": "physical-well-being",
    "98000": "undifferentiated-goods-producing",
    "none": "none",
}


def load_frame(zip_path: Path) -> pd.DataFrame:
    profiles: dict[str, dict] = {}
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("data/training_panel.csv") as handle:
            panel = pd.read_csv(handle)
        for name in archive.namelist():
            if not name.startswith("data/cache/") or not name.endswith(".json"):
                continue
            payload = json.loads(archive.read(name))
            profiles[str(payload["company_number"])] = payload["profile"]

    missing = sorted(set(panel["company_number"].astype(str)) - set(profiles))
    if missing:
        raise RuntimeError(f"No cached profile for {len(missing)} companies, e.g. {missing[:5]}")

    def sic_at(company_number: str, position: int) -> str:
        codes = profiles[company_number].get("sic_codes") or []
        if position < len(codes):
            return str(codes[position])
        return "none"

    company_number = panel["company_number"].astype(str)
    frame = pd.DataFrame(
        {
            "company_number": company_number,
            "as_of_date": panel["as_of_date"].astype(str),
            "final_outcome": panel["final_outcome"].astype(str),
            "age_years": panel["age_years"],
            "accounts_filings": panel["accounts_filings"],
            "avg_days_late_accounts": panel["avg_days_late_accounts"],
            "max_days_late_accounts": panel["max_days_late_accounts"],
            "days_since_last_accounts": panel["days_since_last_accounts"],
            "directors_active": panel["directors_active"],
            "director_resignations_12m": panel["director_resignations_12m"],
            "director_appointments_12m": panel["director_appointments_12m"],
            "address_changes": panel["address_changes"],
            "strike_off_warnings": panel["strike_off_warnings"],
            "charges_created": panel["charges_created"],
            "charges_outstanding": panel["charges_outstanding"],
            "filings_total": panel["filings_total"],
            "n_sic_codes": company_number.map(
                lambda cn: len(profiles[cn].get("sic_codes") or [])
            ),
            "sic_1": company_number.map(lambda cn: sic_at(cn, 0)),
            "sic_2": company_number.map(lambda cn: sic_at(cn, 1)),
            "sic_3": company_number.map(lambda cn: sic_at(cn, 2)),
            "sic_4": company_number.map(lambda cn: sic_at(cn, 3)),
            "jurisdiction": company_number.map(
                lambda cn: str(profiles[cn]["jurisdiction"])
            ),
            "last_accounts_type": panel["accounts_type_last"]
            .fillna("no_accounts")
            .replace("", "no_accounts")
            .astype(str),
        }
    )
    # Panel primary SIC is the historical snapshot value. It matches the
    # profile list for every company; keep the panel value as position 1.
    frame["sic_1"] = panel["sic_1"].astype(str)
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def make_pipeline() -> Pipeline:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC),
            ("cat", encoder, CATEGORICAL),
        ]
    )
    tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    return Pipeline([("preprocess", preprocess), ("tree", tree)])


def feature_names(pipeline: Pipeline) -> list[str]:
    preprocess = pipeline.named_steps["preprocess"]
    raw = list(preprocess.get_feature_names_out())
    names = []
    for name in raw:
        if name.startswith("num__"):
            names.append(name.removeprefix("num__"))
            continue
        if name.startswith("cat__"):
            # One-hot names are column_value. Match the known column so
            # values that themselves contain underscores stay intact.
            matched = False
            for column in CATEGORICAL:
                prefix = f"{column}_"
                body = name.removeprefix("cat__")
                if body.startswith(prefix):
                    value = body[len(prefix) :]
                    value = SIC_NAME.get(value, value)
                    names.append(f"{column}={value}")
                    matched = True
                    break
            if not matched:
                names.append(name.removeprefix("cat__"))
            continue
        names.append(name)
    return names


def score_block(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    per_precision = precision_score(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    per_recall = recall_score(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    true_fail = (y_true.to_numpy() != "active").astype(int)
    pred_fail = (np.asarray(y_pred) != "active").astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(
                y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
            )
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
        ),
        "precision_per_class": {label: float(value) for label, value in zip(LABELS, per_precision)},
        "recall_per_class": {label: float(value) for label, value in zip(LABELS, per_recall)},
        "support_per_class": {
            label: int((y_true == label).sum()) for label in LABELS
        },
        "confusion_matrix": matrix,
        "confusion_matrix_labels": LABELS,
        "not_active": {
            "definition": "Any final_outcome other than active counts as not staying active.",
            "accuracy": float(accuracy_score(true_fail, pred_fail)),
            "precision": float(precision_score(true_fail, pred_fail, zero_division=0)),
            "recall": float(recall_score(true_fail, pred_fail, zero_division=0)),
            "positives": int(true_fail.sum()),
        },
    }


def aggregated_importance(pipeline: Pipeline) -> list[dict]:
    names = feature_names(pipeline)
    importances = pipeline.named_steps["tree"].feature_importances_
    by_source: dict[str, float] = {}
    detailed = []
    for name, importance in zip(names, importances):
        if importance <= 0:
            continue
        source = name.split("=", 1)[0]
        by_source[source] = by_source.get(source, 0.0) + float(importance)
        detailed.append({"feature": name, "importance": float(importance)})
    detailed.sort(key=lambda item: item["importance"], reverse=True)
    grouped = [
        {"feature": feature, "importance": importance}
        for feature, importance in sorted(by_source.items(), key=lambda item: -item[1])
    ]
    return {"by_original_feature": grouped, "by_encoded_column": detailed[:25]}


def fit_search(x_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    search = GridSearchCV(
        estimator=make_pipeline(),
        param_grid={
            "tree__max_depth": [2, 3, 4, 5, 6, 8, 12, None],
            "tree__min_samples_leaf": [5, 10, 20, 40],
            "tree__min_samples_split": [10, 20, 40],
            "tree__criterion": ["gini", "entropy"],
            "tree__class_weight": [None, "balanced"],
        },
        scoring={
            "f1_macro": "f1_macro",
            "accuracy": "accuracy",
        },
        refit="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return search


def params_for_best_accuracy(search: GridSearchCV) -> dict:
    scores = search.cv_results_["mean_test_accuracy"]
    index = int(np.nanargmax(scores))
    params = {
        key.removeprefix("tree__"): value
        for key, value in search.cv_results_["params"][index].items()
    }
    return {
        "params": params,
        "mean_cv_accuracy": float(scores[index]),
        "mean_cv_f1_macro": float(search.cv_results_["mean_test_f1_macro"][index]),
    }


def company_holdout(frame: pd.DataFrame, params: dict) -> dict:
    """Same tree settings, but every snapshot of a company stays on one side."""
    companies = (
        frame.groupby("company_number", as_index=False)["final_outcome"].first()
    )
    train_companies, test_companies = train_test_split(
        companies["company_number"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=companies["final_outcome"],
    )
    train_mask = frame["company_number"].isin(set(train_companies))
    test_mask = frame["company_number"].isin(set(test_companies))
    x_train = frame.loc[train_mask, NUMERIC + CATEGORICAL]
    y_train = frame.loc[train_mask, "final_outcome"]
    x_test = frame.loc[test_mask, NUMERIC + CATEGORICAL]
    y_test = frame.loc[test_mask, "final_outcome"]

    pipeline = make_pipeline()
    pipeline.set_params(**{f"tree__{key}": value for key, value in params.items()})
    pipeline.fit(x_train, y_train)
    train_pred = pipeline.predict(x_train)
    test_pred = pipeline.predict(x_test)
    majority = y_train.mode().iloc[0]
    baseline_pred = np.full(len(y_test), majority)
    return {
        "note": (
            "Companies, not rows, were split. Hyperparameters are the ones "
            "chosen on the row-level training set. This is a check on how much "
            "the row split was helped by seeing the same company twice."
        ),
        "train_companies": int(train_companies.nunique()),
        "test_companies": int(test_companies.nunique()),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "majority_class": str(majority),
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "baseline_test": score_block(y_test, baseline_pred),
        "v1_test": score_block(y_test, test_pred),
    }


def main() -> None:
    frame = load_frame(DATA_ZIP)
    features = frame[NUMERIC + CATEGORICAL]
    target = frame["final_outcome"]

    x_train, x_test, y_train, y_test, index_train, index_test = train_test_split(
        features,
        target,
        frame.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    train_companies = set(frame.loc[index_train, "company_number"])
    test_companies = frame.loc[index_test, "company_number"]
    overlap_rows = int(test_companies.isin(train_companies).sum())

    search = fit_search(x_train, y_train)
    model = search.best_estimator_
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    majority = str(y_train.mode().iloc[0])
    baseline_pred = np.full(len(y_test), majority)

    tree = model.named_steps["tree"]
    names = feature_names(model)
    tree_text = export_text(tree, feature_names=names, decimals=2, max_depth=6)

    chosen = {
        key.removeprefix("tree__"): value for key, value in search.best_params_.items()
    }
    results = {
        "n_rows": int(len(frame)),
        "n_companies": int(frame["company_number"].nunique()),
        "rows_per_company": {
            "min": int(frame.groupby("company_number").size().min()),
            "max": int(frame.groupby("company_number").size().max()),
        },
        "class_counts_rows": target.value_counts().to_dict(),
        "class_counts_companies": (
            frame.groupby("company_number")["final_outcome"].first().value_counts().to_dict()
        ),
        "missing_accounts_lateness_rows": int(frame["avg_days_late_accounts"].isna().sum()),
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified_on": "final_outcome",
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "test_rows_whose_company_is_also_in_train": overlap_rows,
            "share_of_test_rows_with_company_in_train": overlap_rows / len(y_test),
        },
        "optimization": {
            "cv_folds": 5,
            "selection_metric": "f1_macro",
            "reason": (
                "Accuracy is mostly the share of active companies. Macro F1 "
                "asks the tree to separate all four outcomes, including the "
                "failure modes the product is meant to flag."
            ),
            "best_params": chosen,
            "best_mean_cv_f1_macro": float(search.best_score_),
            "best_accuracy_point_in_the_same_grid": params_for_best_accuracy(search),
            "fitted_depth": int(tree.get_depth()),
            "fitted_leaves": int(tree.get_n_leaves()),
        },
        "baseline_majority_class": majority,
        "baseline_test": score_block(y_test, baseline_pred),
        "v1_train": score_block(y_train, train_pred),
        "v1_test": score_block(y_test, test_pred),
        "importance": aggregated_importance(model),
        "tree_text": tree_text,
        "company_holdout": company_holdout(frame, chosen),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    test = results["v1_test"]
    base = results["baseline_test"]
    print(f"rows {results['n_rows']} companies {results['n_companies']}")
    print(f"train {results['split']['train_rows']} test {results['split']['test_rows']}")
    print(
        "test rows whose company is also in train: "
        f"{overlap_rows} ({results['split']['share_of_test_rows_with_company_in_train']:.1%})"
    )
    print("best params", chosen)
    print(f"cv macro F1 {search.best_score_:.3f}")
    print(
        f"accuracy train {results['v1_train']['accuracy']:.3f} "
        f"test {test['accuracy']:.3f} baseline {base['accuracy']:.3f}"
    )
    print(
        f"macro precision test {test['precision_macro']:.3f} "
        f"baseline {base['precision_macro']:.3f}"
    )
    print(
        f"macro recall test {test['recall_macro']:.3f} "
        f"baseline {base['recall_macro']:.3f}"
    )
    print("not-active precision/recall", test["not_active"])
    print("confusion (rows true, cols pred)", LABELS)
    for row in test["confusion_matrix"]:
        print(row)
    print(tree_text)
    holdout = results["company_holdout"]
    print(
        "company holdout accuracy "
        f"v1 {holdout['v1_test']['accuracy']:.3f} "
        f"baseline {holdout['baseline_test']['accuracy']:.3f} "
        f"macro F1 {holdout['v1_test']['f1_macro']:.3f}"
    )
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
