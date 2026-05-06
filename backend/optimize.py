"""
optimize.py --- Data-driven weight optimization for PhishScan
===========================================================
Runs on data/features.csv and produces:
  - Statistical feature analysis (correlation, mutual information)
  - Data-driven weights from Logistic Regression, Random Forest, Gradient Boosting
  - Optimal classification threshold (ROC + Precision-Recall)
  - Evaluation metrics for each model
  - results/optimization_report.txt  (human-readable summary)
  - results/optimized_weights.json   (machine-readable weights for app.py)
  - results/opt_roc.png, results/opt_pr.png, results/opt_importance.png

Run from the project root:
  python backend/optimize.py
"""

import json, os, warnings
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection      import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model         import LogisticRegression
from sklearn.ensemble             import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing        import StandardScaler
from sklearn.feature_selection    import mutual_info_classif
from sklearn.metrics              import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix
)
warnings.filterwarnings('ignore')

# ------ Paths ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DATA_PATH    = 'data/features.csv'
RESULTS_DIR  = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

REPORT_PATH  = os.path.join(RESULTS_DIR, 'optimization_report.txt')
WEIGHTS_PATH = os.path.join(RESULTS_DIR, 'optimized_weights.json')

# ------ Feature columns present in the dataset ---------------------------------------------------------------------------------------------------------------
# NOTE: 5 features (brand_impersonation, link_text_mismatch, form_presence,
#       subdomain_depth, domain_age, ssl_validity) are runtime-computed and NOT
#       in features.csv. Their weights remain literature-based and are noted
#       separately in the output.
DATASET_FEATURES = [
    'url_count', 'has_ip_url', 'has_short_url', 'urgent_keyword',
    'is_free_email', 'subject_urgent', 'body_length',
    'has_html', 'html_text_ratio', 'urls'
]

# ------ Load data ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("=" * 60)
print("  PhishScan --- Weight Optimization Script")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[DATA] Loaded {len(df):,} samples from {DATA_PATH}")
print(f"[DATA] Class distribution:\n{df['label'].value_counts().to_string()}")

# Keep only columns that exist in this dataset
available = [c for c in DATASET_FEATURES if c in df.columns]
missing   = [c for c in DATASET_FEATURES if c not in df.columns]
if missing:
    print(f"[WARN] Columns not found in CSV (skipped): {missing}")

X = df[available].copy()
y = df['label'].copy()

print(f"\n[DATA] Features used for analysis: {available}")
print(f"[DATA] Shape: {X.shape}")

# ------ 1. Statistical Analysis ------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\n" + "---" * 60)
print("  1. STATISTICAL FEATURE ANALYSIS")
print("---" * 60)

# 1a. Correlation with label
corr = X.corrwith(y).abs().sort_values(ascending=False)
print("\n[CORR] Pearson correlation with label (absolute):")
for feat, val in corr.items():
    print(f"  {feat:<20} {val:.4f}")

# 1b. Mutual Information
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi_scores, index=available).sort_values(ascending=False)
print("\n[MI] Mutual Information scores:")
for feat, val in mi_series.items():
    print(f"  {feat:<20} {val:.4f}")

# 1c. Correlation matrix heatmap
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = X.copy()
corr_matrix['label'] = y
sns.heatmap(corr_matrix.corr(), annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, ax=ax, linewidths=0.5)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'opt_correlation.png'), dpi=120)
plt.close()
print("\n[OK] opt_correlation.png saved")

# ------ 2. Train / Test Split ------------------------------------------------------------------------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n[SPLIT] Train: {len(X_train):,}  |  Test: {len(X_test):,}")

scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ------ 3. Model Training & Feature Importances ------------------------------------------------------------------------------------------------------------
print("\n" + "---" * 60)
print("  2. MODEL TRAINING & WEIGHT EXTRACTION")
print("---" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=42),
}

results      = {}
importances  = {}
best_thresh  = {}

for name, m in models.items():
    print(f"\n[TRAIN] {name}...")

    if name == 'Logistic Regression':
        m.fit(X_train_s, y_train)
        y_proba = m.predict_proba(X_test_s)[:, 1]
        imp = np.abs(m.coef_[0])            # absolute coefficients
        imp = imp / imp.sum()               # normalize to [0,1]
    else:
        m.fit(X_train, y_train)
        y_proba = m.predict_proba(X_test)[:, 1]
        imp = m.feature_importances_

    importances[name] = dict(zip(available, imp))

    # Default threshold metrics
    y_pred = (y_proba >= 0.5).astype(int)
    results[name] = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred, zero_division=0),
        'auc':       roc_auc_score(y_test, y_proba),
        'y_proba':   y_proba,
    }

    # Cross-validation
    cv_model = m.__class__(**m.get_params())
    cv_scores = cross_validate(
        cv_model,
        X_train_s if name == 'Logistic Regression' else X_train,
        y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=['accuracy', 'f1', 'roc_auc']
    )
    results[name]['cv_accuracy'] = cv_scores['test_accuracy'].mean()
    results[name]['cv_f1']       = cv_scores['test_f1'].mean()
    results[name]['cv_auc']      = cv_scores['test_roc_auc'].mean()

    print(f"  Accuracy:  {results[name]['accuracy']:.4f}  (CV: {results[name]['cv_accuracy']:.4f})")
    print(f"  Precision: {results[name]['precision']:.4f}")
    print(f"  Recall:    {results[name]['recall']:.4f}")
    print(f"  F1-Score:  {results[name]['f1']:.4f}  (CV: {results[name]['cv_f1']:.4f})")
    print(f"  ROC-AUC:   {results[name]['auc']:.4f}  (CV: {results[name]['cv_auc']:.4f})")
    print(f"\n  Feature importances (normalized):")
    for feat, val in sorted(importances[name].items(), key=lambda x: -x[1]):
        bar = '---' * int(val * 30)
        print(f"    {feat:<20} {val:.4f}  {bar}")

# ------ 4. Threshold Optimization ------------------------------------------------------------------------------------------------------------------------------------------------------
print("\n" + "---" * 60)
print("  3. THRESHOLD OPTIMIZATION")
print("---" * 60)

# Use best model by AUC for threshold optimization
best_name  = max(results, key=lambda k: results[k]['auc'])
best_proba = results[best_name]['y_proba']
print(f"\n[THRESH] Using best model: {best_name} (AUC={results[best_name]['auc']:.4f})")

# ROC curve --- find threshold maximizing F1
fpr, tpr, roc_thresholds = roc_curve(y_test, best_proba)
f1_at_thresh = []
for t in roc_thresholds:
    yp = (best_proba >= t).astype(int)
    f1_at_thresh.append(f1_score(y_test, yp, zero_division=0))

opt_idx    = int(np.argmax(f1_at_thresh))
opt_thresh = float(roc_thresholds[opt_idx])
opt_f1     = float(f1_at_thresh[opt_idx])
print(f"[THRESH] Optimal threshold (max F1):  {opt_thresh:.3f}  (F1={opt_f1:.4f})")

# Precision-Recall curve
prec_arr, rec_arr, pr_thresholds = precision_recall_curve(y_test, best_proba)
f1_pr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
opt_pr_idx    = int(np.argmax(f1_pr))
opt_pr_thresh = float(pr_thresholds[min(opt_pr_idx, len(pr_thresholds)-1)])
print(f"[THRESH] Optimal threshold (PR curve): {opt_pr_thresh:.3f}  (F1={float(f1_pr[opt_pr_idx]):.4f})")

final_thresh = round(float((opt_thresh + opt_pr_thresh) / 2), 3)
print(f"[THRESH] Final recommended threshold:  {final_thresh}")

# Plot ROC
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(fpr, tpr, color='#00ff41', lw=2,
             label=f'{best_name} (AUC={results[best_name]["auc"]:.3f})')
axes[0].axvline(fpr[opt_idx], color='#ff3131', ls='--',
                label=f'Opt threshold={opt_thresh:.3f}')
axes[0].plot([0,1],[0,1],'k--', lw=1)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()
axes[0].set_facecolor('#0a0a0a')
axes[0].tick_params(colors='white')

# Plot PR
axes[1].plot(rec_arr, prec_arr, color='#00ff41', lw=2)
axes[1].axvline(rec_arr[opt_pr_idx], color='#ff3131', ls='--',
                label=f'Opt threshold={opt_pr_thresh:.3f}')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend()
axes[1].set_facecolor('#0a0a0a')
axes[1].tick_params(colors='white')

for ax in axes:
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')

fig.patch.set_facecolor('#0d0d0d')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'opt_roc_pr.png'), dpi=120, facecolor='#0d0d0d')
plt.close()
print("[OK] opt_roc_pr.png saved")

# ------ 5. Feature Importance Comparison Plot ------------------------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#0d0d0d')

for ax, (name, imp) in zip(axes, importances.items()):
    sorted_imp = sorted(imp.items(), key=lambda x: x[1])
    feats = [f for f, _ in sorted_imp]
    vals  = [v for _, v in sorted_imp]
    colors = ['#ff3131' if v > np.percentile(vals, 66) else
              '#ffaa00' if v > np.percentile(vals, 33) else '#00ff41'
              for v in vals]
    ax.barh(feats, vals, color=colors)
    ax.set_title(name, color='white')
    ax.set_facecolor('#0a0a0a')
    ax.tick_params(colors='white', labelsize=8)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Feature Importance Comparison', color='white', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'opt_importance.png'), dpi=120, facecolor='#0d0d0d')
plt.close()
print("[OK] opt_importance.png saved")

# ------ 6. Compute Ensemble Weights ------------------------------------------------------------------------------------------------------------------------------------------------
# Ensemble: average normalized importances across all 3 models
print("\n" + "---" * 60)
print("  4. OPTIMIZED DATA-DRIVEN WEIGHTS")
print("---" * 60)

ensemble = {}
for feat in available:
    vals = [importances[m][feat] for m in models]
    ensemble[feat] = round(float(np.mean(vals)), 6)

# Normalize to sum to 1
total = sum(ensemble.values())
ensemble_norm = {k: round(v / total, 6) for k, v in ensemble.items()}

# Scale to max_score space for app.py compatibility
# Original max scores from app.py design doc (dataset features only):
DESIGN_MAX = {
    'url_count':      12,
    'has_ip_url':     15,
    'has_short_url':  12,
    'urgent_keyword': 15,
    'is_free_email':  10,
    'subject_urgent':  7,
    'body_length':     8,
    'has_html':        4,
    'html_text_ratio': 8,
    'urls':            4,   # urls is binary, low max
}

# Data-driven max scores: rescale proportionally to [4, 15]
min_imp = min(ensemble_norm.values())
max_imp = max(ensemble_norm.values())

def scale_to_range(v, lo=4, hi=15):
    if max_imp == min_imp: return (lo + hi) // 2
    return round(lo + (v - min_imp) / (max_imp - min_imp) * (hi - lo))

data_driven_max = {
    feat: scale_to_range(ensemble_norm[feat])
    for feat in available
}

print("\n[WEIGHTS] Feature ranking (ensemble of LR + RF + GB):")
ranked = sorted(ensemble_norm.items(), key=lambda x: -x[1])
for feat, imp in ranked:
    old = DESIGN_MAX.get(feat, '?')
    new = data_driven_max[feat]
    bar = '---' * int(imp * 60)
    print(f"  {feat:<20} importance={imp:.4f}  old_max={old:>2}  new_max={new:>2}  {bar}")

# Also identify low-importance features
LOW_THRESHOLD = 0.05
low_imp = [f for f, v in ensemble_norm.items() if v < LOW_THRESHOLD]
if low_imp:
    print(f"\n[WARN] Low-importance features (< {LOW_THRESHOLD}): {low_imp}")
    print("       Consider removing or reducing their weight.")

# ------ 7. Best Model Metrics at Optimal Threshold ---------------------------------------------------------------------------------------------------
y_opt = (best_proba >= final_thresh).astype(int)
print(f"\n[METRICS] {best_name} at threshold={final_thresh}:")
print(f"  Accuracy:  {accuracy_score(y_test, y_opt):.4f}")
print(f"  Precision: {precision_score(y_test, y_opt, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test, y_opt, zero_division=0):.4f}")
print(f"  F1-Score:  {f1_score(y_test, y_opt, zero_division=0):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, best_proba):.4f}")
print(f"\n{classification_report(y_test, y_opt, target_names=['Legitimate','Phishing'])}")

# Confusion matrix
cm  = confusion_matrix(y_test, y_opt)
tn, fp, fn, tp = cm.ravel()
print(f"  True Negatives (correct legit):  {tn}")
print(f"  False Positives (false alarm):   {fp}")
print(f"  False Negatives (missed phish):  {fn}")
print(f"  True Positives (caught phish):   {tp}")

# ------ 8. Save optimized weights JSON ------------------------------------------------------------------------------------------------------------------------------------
output = {
    'best_model':        best_name,
    'optimal_threshold': final_thresh,
    'dataset_features':  available,
    'ensemble_importance': {k: round(v, 6) for k, v in ensemble_norm.items()},
    'data_driven_max_scores': data_driven_max,
    'low_importance_features': low_imp,
    'metrics': {
        name: {
            k: round(v, 4) for k, v in res.items()
            if isinstance(v, float)
        }
        for name, res in results.items()
    },
    'note': (
        "Features not in dataset (brand_impersonation, link_text_mismatch, "
        "form_presence, subdomain_depth, domain_age, ssl_validity) retain "
        "literature-based weights from the PFE design document."
    )
}

with open(WEIGHTS_PATH, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n[OK] Optimized weights saved to {WEIGHTS_PATH}")

# ------ 9. Text Report ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
report_lines = [
    "=" * 60,
    "  PhishScan --- Optimization Report",
    "=" * 60,
    "",
    f"Dataset: {DATA_PATH}",
    f"Samples: {len(df):,}  |  Features: {len(available)}",
    f"Class distribution: {dict(df['label'].value_counts())}",
    "",
    "---" * 60,
    "  MODEL COMPARISON (5-fold cross-validation)",
    "---" * 60,
]
for name, res in results.items():
    report_lines += [
        f"\n{name}:",
        f"  Accuracy  (CV): {res['cv_accuracy']:.4f}",
        f"  F1-Score  (CV): {res['cv_f1']:.4f}",
        f"  ROC-AUC   (CV): {res['cv_auc']:.4f}",
    ]
report_lines += [
    "",
    "---" * 60,
    "  BEST MODEL & THRESHOLD",
    "---" * 60,
    f"  Best model:          {best_name}",
    f"  Optimal threshold:   {final_thresh}",
    f"  F1 at threshold:     {f1_score(y_test, y_opt, zero_division=0):.4f}",
    f"  Recall at threshold: {recall_score(y_test, y_opt, zero_division=0):.4f}",
    "",
    "---" * 60,
    "  DATA-DRIVEN FEATURE WEIGHTS",
    "---" * 60,
]
for feat, imp in sorted(ensemble_norm.items(), key=lambda x: -x[1]):
    report_lines.append(
        f"  {feat:<20} importance={imp:.4f}  max_score={data_driven_max[feat]}"
    )
if low_imp:
    report_lines += ["", f"  LOW-IMPACT FEATURES: {low_imp}"]
report_lines += [
    "",
    "---" * 60,
    "  RECOMMENDATION",
    "---" * 60,
    "",
    f"  Use {best_name} as the primary classifier.",
    f"  Apply threshold = {final_thresh} (optimized for F1).",
    "  Update app.py MAX_SCORES with the data_driven_max_scores above.",
    "  Features with importance < 0.05 can be removed to reduce complexity.",
    "",
    "  For the 6 runtime features (brand_impersonation, link_text_mismatch,",
    "  form_presence, subdomain_depth, domain_age, ssl_validity), retain",
    "  literature-based weights --- they cannot be validated on this dataset.",
]

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"[OK] Report saved to {REPORT_PATH}")

print("\n" + "=" * 60)
print("  DONE. Files written to results/:")
print("    optimization_report.txt")
print("    optimized_weights.json")
print("    opt_correlation.png")
print("    opt_roc_pr.png")
print("    opt_importance.png")
print("=" * 60)

