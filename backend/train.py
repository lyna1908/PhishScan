import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Load data ─────────────────────────────────────────────────────
df = pd.read_csv('../data/features.csv')
X = df.drop('label', axis=1)
y = df['label']

# ── Split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── Models ────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM':                 SVC(probability=True, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n⏳ Training {name}...")
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        'model':     model,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'auc':       roc_auc_score(y_test, y_proba),
        'y_pred':    y_pred
    }

    print(f"  Accuracy:  {results[name]['accuracy']:.4f}")
    print(f"  Precision: {results[name]['precision']:.4f}")
    print(f"  Recall:    {results[name]['recall']:.4f}")
    print(f"  F1 Score:  {results[name]['f1']:.4f}")
    print(f"  ROC-AUC:   {results[name]['auc']:.4f}")

# ── Confusion matrices ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('../results/confusion_matrices.png')
print("\n✅ confusion_matrices.png saved!")

# ── ROC curves ───────────────────────────────────────────────────
from sklearn.metrics import roc_curve
plt.figure(figsize=(8, 6))
for name, res in results.items():
    y_proba = res['model'].predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.savefig('../results/roc_curves.png')
print("✅ roc_curves.png saved!")

# ── Select and save best model ────────────────────────────────────
best_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_name]['model']
print(f"\n🏆 Best model: {best_name} (F1={results[best_name]['f1']:.4f})")
joblib.dump(best_model, 'model/best_model.pkl')
print("✅ best_model.pkl saved!")