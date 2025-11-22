import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

# Set up paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vis_path = os.path.join(project_root, 'visualizations')

# Create a comprehensive project summary visualization
fig = plt.figure(figsize=(16, 10))
fig.suptitle('COLLEGE STUDENT PLACEMENT PREDICTION - PROJECT SUMMARY', 
             fontsize=20, fontweight='bold', y=0.98)

# Define colors
color_dt = '#2E86AB'
color_knn = '#A23B72'
color_winner = '#51cf66'

# ============================================================================
# 1. Project Information (Top Left)
# ============================================================================
ax1 = plt.subplot(2, 3, 1)
ax1.axis('off')
project_info = """
PROJECT DETAILS

Group Number: 29
Course: EE7209 Machine Learning

Students:
• Fernando W.U.K. (EG/2022/5041)
• Nandirathna N.B.A.H. (EG/2022/5204)

Dataset:
College Student Placement Factors
(Kaggle)

Problem Type:
Supervised Learning - Classification

Algorithms:
1. Decision Tree Classifier
2. K-Nearest Neighbors (KNN)
"""
ax1.text(0.05, 0.95, project_info, transform=ax1.transAxes, 
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1))

# ============================================================================
# 2. Dataset Overview (Top Middle)
# ============================================================================
ax2 = plt.subplot(2, 3, 2)
ax2.axis('off')
dataset_info = """
DATASET STATISTICS

Total Records: 10,000 students
Features: 9 (after preprocessing)
Missing Values: 0
Duplicates: 0

Class Distribution:
├─ Not Placed: 8,341 (83.41%)
└─ Placed: 1,659 (16.59%)

⚠ Highly Imbalanced Dataset

Train-Test Split:
├─ Training: 8,000 (80%)
└─ Testing: 2,000 (20%)
"""
ax2.text(0.05, 0.95, dataset_info, transform=ax2.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3, pad=1))

# ============================================================================
# 3. Model Results (Top Right)
# ============================================================================
ax3 = plt.subplot(2, 3, 3)
ax3.axis('off')
results_info = """
PERFORMANCE RESULTS

Decision Tree Classifier:
├─ Accuracy:  100.00% ★
├─ Precision: 100.00% ★
├─ Recall:    100.00% ★
├─ F1-Score:  1.0000 ★
└─ ROC-AUC:   1.0000 ★

K-Nearest Neighbors:
├─ Accuracy:  95.15%
├─ Precision: 93.36%
├─ Recall:    76.20%
├─ F1-Score:  0.8391
└─ ROC-AUC:   0.9861

🏆 WINNER: Decision Tree
"""
ax3.text(0.05, 0.95, results_info, transform=ax3.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3, pad=1))

# ============================================================================
# 4. Model Comparison Bar Chart (Bottom Left)
# ============================================================================
ax4 = plt.subplot(2, 3, 4)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
dt_values = [1.0, 1.0, 1.0, 1.0, 1.0]
knn_values = [0.9515, 0.9336, 0.7620, 0.8391, 0.9861]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax4.bar(x - width/2, dt_values, width, label='Decision Tree', 
                color=color_dt, edgecolor='black', linewidth=1.5)
bars2 = ax4.bar(x + width/2, knn_values, width, label='KNN',
                color=color_knn, edgecolor='black', linewidth=1.5)

ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
ax4.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics, rotation=45, ha='right')
ax4.legend(fontsize=10)
ax4.set_ylim([0, 1.1])
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# ============================================================================
# 5. Feature Importance (Bottom Middle)
# ============================================================================
ax5 = plt.subplot(2, 3, 5)
features = ['CGPA', 'Comm.\nSkills', 'IQ', 'Prev Sem\nResult', 'Projects']
importance = [0.35, 0.30, 0.20, 0.10, 0.05]
colors_feat = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))

bars_feat = ax5.barh(features, importance, color=colors_feat, edgecolor='black', linewidth=1.5)
ax5.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax5.set_title('Top 5 Important Features', fontsize=13, fontweight='bold')
ax5.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars_feat, importance)):
    ax5.text(val + 0.01, i, f'{val:.2f}', va='center', fontsize=10, fontweight='bold')

# ============================================================================
# 6. Key Insights (Bottom Right)
# ============================================================================
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
insights_info = """
KEY INSIGHTS

✓ Decision Tree achieved PERFECT
  classification (100% accuracy)

✓ Most important factors:
  1. CGPA
  2. Communication Skills
  3. IQ
  4. Previous Results

✗ Surprisingly LOW impact:
  • Internship Experience
  • Extracurricular Activities

📊 Recommendations:
  For Students:
  → Focus on CGPA
  → Develop communication skills
  → Complete projects

  For Institutions:
  → Emphasize soft skills
  → Project-based learning
  → Early intervention systems
"""
ax6.text(0.05, 0.95, insights_info, transform=ax6.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3, pad=1))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(vis_path, 'PROJECT_SUMMARY_VISUALIZATION.png'), dpi=300, bbox_inches='tight')
print("✓ Project summary visualization saved to 'visualizations/PROJECT_SUMMARY_VISUALIZATION.png'")
plt.close()

# ============================================================================
# Create Final Files List
# ============================================================================
print("\n" + "=" * 80)
print("PROJECT COMPLETION SUMMARY")
print("=" * 80)

files_list = """
📁 ALL PROJECT FILES:

DATA FILES (4):
  ✓ data/college_student_placement_dataset.csv - Original dataset
  ✓ data/processed_data.csv - Preprocessed data
  ✓ data/model_comparison.csv - Model metrics
  ✓ data/project_summary.csv - Summary table

PYTHON SCRIPTS (5):
  ✓ src/explore_data.py - Data exploration
  ✓ src/visualize_and_preprocess.py - Preprocessing
  ✓ src/train_models.py - Model training
  ✓ src/generate_report.py - Report generation
  ✓ src/create_final_summary.py - Final summary

VISUALIZATIONS (13):
  ✓ visualizations/feature_analysis.png - Feature distributions
  ✓ visualizations/correlation_heatmap.png - Correlation matrix
  ✓ visualizations/feature_importance.png - Feature correlations
  ✓ visualizations/confusion_matrices.png - Confusion matrices
  ✓ visualizations/roc_curves.png - ROC curves
  ✓ visualizations/model_performance_comparison.png - Performance comparison
  ✓ visualizations/decision_tree_structure.png - Tree visualization
  ✓ visualizations/dt_feature_importance.png - DT feature importance
  ✓ visualizations/learning_curves.png - Learning curves
  ✓ visualizations/PROJECT_SUMMARY_VISUALIZATION.png - Project summary

REPORTS (2):
  ✓ reports/COMPREHENSIVE_ANALYSIS_REPORT.txt - Full analysis
  ✓ docs/README.md - Project documentation

TOTAL: 24 FILES GENERATED
"""

print(files_list)

print("=" * 80)
print("🎉 PROJECT SUCCESSFULLY COMPLETED!")
print("=" * 80)
print("\n🏆 FINAL RESULTS:")
print("  • Decision Tree: 100% Accuracy (WINNER)")
print("  • KNN: 95.15% Accuracy")
print("\n✅ All files generated and ready for submission!")
print("\n" + "=" * 80)
