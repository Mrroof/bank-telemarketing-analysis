from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
# information into a pre-call targeting model.
features = ["age", "job", "account_balance", "campaign_contacts",
            "previous_contacts", "contact_type"]
num = ["age", "account_balance", "campaign_contacts", "previous_contacts"]
cat = ["job", "contact_type"]
prep = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), num),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat)
])
model = Pipeline([
    ("prep", prep),
    ("logit", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

# Keep the test set untouched. Select a decision threshold on a validation
# split from the training data, maximizing validation F1.
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df.subscribed_flag, test_size=0.25,
    random_state=42, stratify=df.subscribed_flag
)
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.25,
    random_state=21, stratify=y_train
)
model.fit(X_fit, y_fit)
val_scores = model.predict_proba(X_val)[:, 1]
thresholds = [i / 100 for i in range(10, 81, 5)]
threshold = max(thresholds, key=lambda t: f1_score(y_val, (val_scores >= t).astype(int), zero_division=0))

# Refit on all training data after selecting the threshold.
model.fit(X_train, y_train)
scores = model.predict_proba(X_test)[:, 1]
pred = (scores >= threshold).astype(int)
report = classification_report(y_test, pred, zero_division=0)
auc = roc_auc_score(y_test, scores)
ap = average_precision_score(y_test, scores)
cm = confusion_matrix(y_test, pred)

(out / "classification_report.txt").write_text(
    "SYNTHETIC DATA ONLY\n"
    "Call duration excluded to avoid post-call leakage.\n"
    f"Validation-selected threshold: {threshold:.2f}\n"
    f"Test ROC AUC: {auc:.3f}\nTest average precision: {ap:.3f}\n"
    f"Confusion matrix [[TN, FP], [FN, TP]]: {cm.tolist()}\n\n{report}",
    encoding="utf-8"
)
pd.DataFrame([{
    "threshold": threshold, "roc_auc": auc, "average_precision": ap,
    "precision_subscribers": cm[1,1] / max(cm[:,1].sum(), 1),
    "recall_subscribers": cm[1,1] / max(cm[1,:].sum(), 1),
    "f1_subscribers": f1_score(y_test, pred, zero_division=0)
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

print(report)
print("ROC AUC:", round(auc, 3), "Average precision:", round(ap, 3))
print("Validation-selected threshold:", threshold)
print("Confusion matrix [[TN, FP], [FN, TP]]:", cm.tolist())
