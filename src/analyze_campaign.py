from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/telemarketing_demo.csv")
out = ROOT / "outputs"
out.mkdir(exist_ok=True)

target = "term_deposit_subscribed"
df["subscribed_flag"] = (df[target].str.lower() == "yes").astype(int)

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

features = [
    "age", "job", "account_balance", "campaign_contacts",
    "previous_contacts", "contact_type"
]
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
    ("logit", LogisticRegression(max_iter=1000))
])

Xtr, Xte, ytr, yte = train_test_split(
    df[features], df.subscribed_flag, test_size=0.25,
    random_state=42, stratify=df.subscribed_flag
)
model.fit(Xtr, ytr)
pred = model.predict(Xte)
scores = model.predict_proba(Xte)[:, 1]

report = classification_report(yte, pred, zero_division=0)
auc = roc_auc_score(yte, scores)
(out / "classification_report.txt").write_text(
    "SYNTHETIC DATA ONLY\n" + report + f"\nROC AUC: {auc:.3f}\n",
    encoding="utf-8"
)
print(report, "ROC AUC:", round(auc, 3))
