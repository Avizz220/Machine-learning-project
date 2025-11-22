import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve)
import warnings
import os
warnings.filterwarnings('ignore')

# Set up paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, 'data', 'processed_data.csv')
results_path = os.path.join(project_root, 'data', 'model_comparison.csv')
vis_path = os.path.join(project_root, 'visualizations')

# Set style
sns.set_style("whitegrid")

print("=" * 80)
print("MACHINE LEARNING MODEL TRAINING")
print("=" * 80)

# Load processed data
df = pd.read_csv(data_path)
print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Separate features and target
X = df.drop('Placement', axis=1)
y = df['Placement']

print(f"\nFeatures (X): {X.shape}")
print(f"Target (y): {y.shape}")
print(f"\nClass Distribution:")
print(f"  Not Placed (0): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.2f}%)")
print(f"  Placed (1): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.2f}%)")

# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("SPLITTING DATA INTO TRAIN AND TEST SETS")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining Set: {X_train.shape[0]} samples")
print(f"Testing Set: {X_test.shape[0]} samples")
print(f"\nClass distribution in training set:")
print(f"  Not Placed: {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.2f}%)")
print(f"  Placed: {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.2f}%)")

# ============================================================================
# FEATURE SCALING (Required for KNN)
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE SCALING FOR KNN")
print("=" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Features scaled using StandardScaler")
print("  (Decision Tree doesn't require scaling, but KNN does)")

# ============================================================================
# MODEL 1: DECISION TREE CLASSIFIER
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 1: DECISION TREE CLASSIFIER")
print("=" * 80)

print("\n[1] Training Decision Tree with default parameters...")
dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_train, y_train)
print("✓ Training completed")

print("\n[2] Performing hyperparameter tuning using GridSearchCV...")
dt_params = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced', None]
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
dt_grid.fit(X_train, y_train)

print(f"\n✓ Best parameters found: {dt_grid.best_params_}")
print(f"✓ Best cross-validation F1 score: {dt_grid.best_score_:.4f}")

dt_best = dt_grid.best_estimator_

# ============================================================================
# MODEL 2: K-NEAREST NEIGHBORS (KNN)
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 2: K-NEAREST NEIGHBORS (KNN)")
print("=" * 80)

print("\n[1] Training KNN with default parameters (k=5)...")
knn_default = KNeighborsClassifier()
knn_default.fit(X_train_scaled, y_train)
print("✓ Training completed")

print("\n[2] Performing hyperparameter tuning using GridSearchCV...")
knn_params = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 19, 25],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]
}

knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    knn_params,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
knn_grid.fit(X_train_scaled, y_train)

print(f"\n✓ Best parameters found: {knn_grid.best_params_}")
print(f"✓ Best cross-validation F1 score: {knn_grid.best_score_:.4f}")

knn_best = knn_grid.best_estimator_

# ============================================================================
# MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("MODEL EVALUATION ON TEST SET")
print("=" * 80)

# Decision Tree Predictions
y_pred_dt = dt_best.predict(X_test)
y_pred_proba_dt = dt_best.predict_proba(X_test)[:, 1]

# KNN Predictions
y_pred_knn = knn_best.predict(X_test_scaled)
y_pred_proba_knn = knn_best.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics for both models
def calculate_metrics(y_true, y_pred, y_pred_proba, model_name):
    print(f"\n{model_name} Performance:")
    print("-" * 60)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }

dt_metrics = calculate_metrics(y_test, y_pred_dt, y_pred_proba_dt, "Decision Tree")
knn_metrics = calculate_metrics(y_test, y_pred_knn, y_pred_proba_knn, "K-Nearest Neighbors")

# Classification Reports
print("\n" + "=" * 80)
print("DETAILED CLASSIFICATION REPORTS")
print("=" * 80)

print("\nDecision Tree Classification Report:")
print("-" * 60)
print(classification_report(y_test, y_pred_dt, target_names=['Not Placed', 'Placed']))

print("\nK-Nearest Neighbors Classification Report:")
print("-" * 60)
print(classification_report(y_test, y_pred_knn, target_names=['Not Placed', 'Placed']))

# ============================================================================
# SAVE RESULTS TO CSV
# ============================================================================
results_df = pd.DataFrame([dt_metrics, knn_metrics])
results_df.to_csv(results_path, index=False)
print("\n✓ Model comparison saved to 'data/model_comparison.csv'")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# 1. Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cm_dt = confusion_matrix(y_test, y_pred_dt)
cm_knn = confusion_matrix(y_test, y_pred_knn)

sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=['Not Placed', 'Placed'], 
            yticklabels=['Not Placed', 'Placed'],
            cbar_kws={'label': 'Count'})
axes[0].set_title('Decision Tree - Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Actual', fontsize=12)
axes[0].set_xlabel('Predicted', fontsize=12)

sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Not Placed', 'Placed'],
            yticklabels=['Not Placed', 'Placed'],
            cbar_kws={'label': 'Count'})
axes[1].set_title('K-Nearest Neighbors - Confusion Matrix', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Actual', fontsize=12)
axes[1].set_xlabel('Predicted', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(vis_path, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
print("✓ Confusion matrices saved to 'visualizations/confusion_matrices.png'")

# 2. ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))

fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_proba_knn)

ax.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_score(y_test, y_pred_proba_dt):.4f})', 
        linewidth=2.5, color='#2E86AB')
ax.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_score(y_test, y_pred_proba_knn):.4f})', 
        linewidth=2.5, color='#A23B72')
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1.5)

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(vis_path, 'roc_curves.png'), dpi=300, bbox_inches='tight')
print("✓ ROC curves saved to 'visualizations/roc_curves.png'")

# 3. Model Performance Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values_dt = [dt_metrics['Accuracy'], dt_metrics['Precision'], 
                     dt_metrics['Recall'], dt_metrics['F1-Score']]
metrics_values_knn = [knn_metrics['Accuracy'], knn_metrics['Precision'],
                      knn_metrics['Recall'], knn_metrics['F1-Score']]

# Bar plots for each metric
for idx, (metric_name, dt_val, knn_val) in enumerate(zip(metrics_names, metrics_values_dt, metrics_values_knn)):
    row = idx // 2
    col = idx % 2
    
    ax = axes[row, col]
    bars = ax.bar(['Decision Tree', 'KNN'], [dt_val, knn_val], 
                  color=['#2E86AB', '#A23B72'], edgecolor='black', linewidth=1.5)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} Comparison', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}\n({height*100:.2f}%)',
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(vis_path, 'model_performance_comparison.png'), dpi=300, bbox_inches='tight')
print("✓ Model performance comparison saved to 'visualizations/model_performance_comparison.png'")

# 4. Decision Tree Visualization
fig, ax = plt.subplots(figsize=(20, 12))
plot_tree(dt_best, filled=True, feature_names=X.columns, 
          class_names=['Not Placed', 'Placed'], 
          rounded=True, fontsize=10, ax=ax)
plt.title('Decision Tree Structure (Optimized)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(vis_path, 'decision_tree_structure.png'), dpi=300, bbox_inches='tight')
print("✓ Decision tree structure saved to 'visualizations/decision_tree_structure.png'")

# 5. Feature Importance (Decision Tree)
fig, ax = plt.subplots(figsize=(10, 8))
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_best.feature_importances_
}).sort_values('Importance', ascending=True)

colors_importance = plt.cm.viridis(feature_importance['Importance'] / feature_importance['Importance'].max())
feature_importance.plot(x='Feature', y='Importance', kind='barh', ax=ax, 
                        color=colors_importance, edgecolor='black', linewidth=1.5, legend=False)
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
ax.set_title('Feature Importance - Decision Tree', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(vis_path, 'dt_feature_importance.png'), dpi=300, bbox_inches='tight')
print("✓ Decision tree feature importance saved to 'visualizations/dt_feature_importance.png'")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)

print("\n📊 FINAL RESULTS SUMMARY:")
print("\nDecision Tree:")
print(f"  Best Parameters: {dt_grid.best_params_}")
print(f"  Test Accuracy: {dt_metrics['Accuracy']:.4f}")
print(f"  Test F1-Score: {dt_metrics['F1-Score']:.4f}")

print("\nK-Nearest Neighbors:")
print(f"  Best Parameters: {knn_grid.best_params_}")
print(f"  Test Accuracy: {knn_metrics['Accuracy']:.4f}")
print(f"  Test F1-Score: {knn_metrics['F1-Score']:.4f}")

if dt_metrics['F1-Score'] > knn_metrics['F1-Score']:
    print(f"\n🏆 Winner: Decision Tree (F1-Score: {dt_metrics['F1-Score']:.4f})")
elif knn_metrics['F1-Score'] > dt_metrics['F1-Score']:
    print(f"\n🏆 Winner: K-Nearest Neighbors (F1-Score: {knn_metrics['F1-Score']:.4f})")
else:
    print(f"\n🤝 Tie: Both models have equal F1-Score: {dt_metrics['F1-Score']:.4f}")

print("\n📁 Generated Files:")
print("  1. data/model_comparison.csv - Detailed metrics comparison")
print("  2. visualizations/confusion_matrices.png - Confusion matrices for both models")
print("  3. visualizations/roc_curves.png - ROC curves comparison")
print("  4. visualizations/model_performance_comparison.png - Performance metrics comparison")
print("  5. visualizations/decision_tree_structure.png - Visual representation of decision tree")
print("  6. visualizations/dt_feature_importance.png - Feature importance from decision tree")

print("\n" + "=" * 80)
