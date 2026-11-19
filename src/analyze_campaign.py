from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix, f1_score
)

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/telemarketing_demo.csv")
out = ROOT / "outputs"
out.mkdir(exist_ok=True)

target = "term_deposit_subscribed"
df["subscribed_flag"] = (df[target].astype(str).str.lower() == "yes").astype(int)
print("Rows:", len(df), "Demo subscription rate:", round(df.subscribed_flag.mean(), 3))
print(df.groupby("contact_type").subscribed_flag.agg(["count", "mean"]))
print(df.groupby("job").subscribed_flag.agg(["count", "mean"]).sort_values("mean", ascending=False))

ax = df.groupby("campaign_contacts").subscribed_flag.mean().plot(
    kind="bar", figsize=(8, 4), title="Demo subscription rate by campaign contacts"
)
ax.set_xlabel("Campaign contacts")
ax.set_ylabel("Subscription rate")
plt.tight_layout()
plt.savefig(out / "subscription_by_contacts.png", dpi=160)
plt.close()

# Exclude call duration: it is only known after the call and would leak
# post-call information into a pre-call targeting model.
features = ["age", "job", "account_balance", "campaign_contacts",
            "previous_contacts", "contact_type"]
num = ["age", "account_balance", "campaign_contacts", "previous_contacts"]
cat = ["job", "contact_type"]

def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), num),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat)
    ])

def make_model(kind):
    classifier = (
        LogisticRegression(max_iter=2000, class_weight="balanced")
        if kind == "logistic"
        else RandomForestClassifier(
            n_estimators=400, min_samples_leaf=8,
            class_weight="balanced_subsample", random_state=42,
            n_jobs=-1
        )
    )
    return Pipeline([("prep", make_preprocessor()), ("model", classifier)])

# Keep the final test set untouched. Compare candidate models on validation
# average precision, then select the decision threshold on validation F1.
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df.subscribed_flag, test_size=0.25,
    random_state=42, stratify=df.subscribed_flag
)
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.25,
    random_state=21, stratify=y_train
)

candidates = {}
for kind in ["logistic", "random_forest"]:
    candidate = make_model(kind)
    candidate.fit(X_fit, y_fit)
    candidates[kind] = {
        "model": candidate,
        "validation_average_precision": average_precision_score(
            y_val, candidate.predict_proba(X_val)[:, 1]
        )
    }

selected_name = max(
    candidates, key=lambda name: candidates[name]["validation_average_precision"]
)
model = make_model(selected_name)
model.fit(X_train, y_train)

val_model = candidates[selected_name]["model"]
val_scores = val_model.predict_proba(X_val)[:, 1]
thresholds = [i / 100 for i in range(10, 81, 5)]
threshold = max(
    thresholds,
    key=lambda t: f1_score(y_val, (val_scores >= t).astype(int), zero_division=0)
)

scores = model.predict_proba(X_test)[:, 1]
pred = (scores >= threshold).astype(int)
report = classification_report(y_test, pred, zero_division=0)
auc = roc_auc_score(y_test, scores)
ap = average_precision_score(y_test, scores)
cm = confusion_matrix(y_test, pred)
precision_value = cm[1, 1] / max(cm[:, 1].sum(), 1)
recall_value = cm[1, 1] / max(cm[1, :].sum(), 1)
f1_value = f1_score(y_test, pred, zero_division=0)

(out / "classification_report.txt").write_text(
    "SYNTHETIC DATA ONLY\n"
    "Call duration excluded to avoid post-call leakage.\n"
    f"Selected model: {selected_name}\n"
    f"Validation average precision - logistic: {candidates['logistic']['validation_average_precision']:.3f}\n"
    f"Validation average precision - random forest: {candidates['random_forest']['validation_average_precision']:.3f}\n"
    f"Validation-selected threshold: {threshold:.2f}\n"
    f"Test ROC AUC: {auc:.3f}\nTest average precision: {ap:.3f}\n"
    f"Test subscriber precision: {precision_value:.3f}\n"
    f"Test subscriber recall: {recall_value:.3f}\n"
    f"Test subscriber F1: {f1_value:.3f}\n"
    f"Confusion matrix [[TN, FP], [FN, TP]]: {cm.tolist()}\n\n{report}",
    encoding="utf-8"
)
pd.DataFrame([{
    "selected_model": selected_name,
    "logistic_validation_average_precision": candidates["logistic"]["validation_average_precision"],
    "random_forest_validation_average_precision": candidates["random_forest"]["validation_average_precision"],
    "threshold": threshold, "roc_auc": auc, "average_precision": ap,
    "precision_subscribers": precision_value,
    "recall_subscribers": recall_value, "f1_subscribers": f1_value
}]).to_csv(out / "model_metrics.csv", index=False)

precision, recall, _ = precision_recall_curve(y_test, scores)
plt.figure(figsize=(7, 4))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-recall curve (demo test set)")
plt.tight_layout()
plt.savefig(out / "precision_recall_curve.png", dpi=160)
plt.close()

print("Selected model:", selected_name)
print("Validation average precision:", {
    name: round(value["validation_average_precision"], 3)
    for name, value in candidates.items()
})
print(report)
print("ROC AUC:", round(auc, 3), "Average precision:", round(ap, 3))
print("Validation-selected threshold:", threshold)
print("Confusion matrix [[TN, FP], [FN, TP]]:", cm.tolist())
