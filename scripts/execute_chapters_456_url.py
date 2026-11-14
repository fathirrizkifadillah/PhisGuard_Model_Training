import nbformat
from nbclient import NotebookClient
from pathlib import Path

nb_path = Path('url.ipynb').resolve()
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

cell_55_code = """# Evaluasi menggunakan Stratified 5-Fold CV pada subset training (sample 25k untuk efisiensi k-fold)
sample_cv_idx = X_train.sample(25000, random_state=42).index
X_cv = X_train.loc[sample_cv_idx]
y_cv = y_train.loc[sample_cv_idx]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1', 'roc_auc': 'roc_auc'}

# Gunakan single-thread estimator selama cross-validation untuk mencegah thread deadlock pada Windows
rf_cv = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=5, random_state=42, n_jobs=1)
xgb_cv = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.08, colsample_bytree=0.8, subsample=0.8, tree_method='hist', random_state=42, n_jobs=1, eval_metric='logloss')

print("Menjalankan Stratified 5-Fold Cross Validation...")
cv_rf = cross_validate(rf_cv, X_cv, y_cv, cv=cv, scoring=scoring, n_jobs=2)
cv_xgb = cross_validate(xgb_cv, X_cv, y_cv, cv=cv, scoring=scoring, n_jobs=2)

cv_comparison = pd.DataFrame({
    'Metric (5-Fold CV Mean)': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Fit Time (s)'],
    'Random Forest (Baseline)': [
        f"{cv_rf['test_accuracy'].mean():.4f} +/- {cv_rf['test_accuracy'].std():.4f}",
        f"{cv_rf['test_precision'].mean():.4f} +/- {cv_rf['test_precision'].std():.4f}",
        f"{cv_rf['test_recall'].mean():.4f} +/- {cv_rf['test_recall'].std():.4f}",
        f"{cv_rf['test_f1'].mean():.4f} +/- {cv_rf['test_f1'].std():.4f}",
        f"{cv_rf['test_roc_auc'].mean():.4f} +/- {cv_rf['test_roc_auc'].std():.4f}",
        f"{cv_rf['fit_time'].mean():.2f}s"
    ],
    'XGBoost (Production)': [
        f"{cv_xgb['test_accuracy'].mean():.4f} +/- {cv_xgb['test_accuracy'].std():.4f}",
        f"{cv_xgb['test_precision'].mean():.4f} +/- {cv_xgb['test_precision'].std():.4f}",
        f"{cv_xgb['test_recall'].mean():.4f} +/- {cv_xgb['test_recall'].std():.4f}",
        f"{cv_xgb['test_f1'].mean():.4f} +/- {cv_xgb['test_f1'].std():.4f}",
        f"{cv_xgb['test_roc_auc'].mean():.4f} +/- {cv_xgb['test_roc_auc'].std():.4f}",
        f"{cv_xgb['fit_time'].mean():.2f}s"
    ]
})

print("Tabel Komparasi Performa Cross-Validation:")
display(cv_comparison)
"""
nb.cells[55].source = cell_55_code

client = NotebookClient(nb, timeout=300, kernel_name='python3')
print("Starting execution of cells 50 to 62...")
with client.setup_kernel():
    for idx, cell in enumerate(nb.cells):
        if idx >= 50 and cell.cell_type == 'code':
            print(f"Executing cell {idx}...")
            client.execute_cell(cell, idx)

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Cells 50 to 62 executed and saved successfully!")
