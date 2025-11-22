import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load the dataset
df = pd.read_csv('college_student_placement_dataset.csv')

print("=" * 80)
print("DATA PREPROCESSING STARTED")
print("=" * 80)

# Create a copy for preprocessing
df_processed = df.copy()

# 1. Handle College_ID (will be dropped as it's just an identifier)
print("\n1. Dropping College_ID (identifier column)...")
df_processed = df_processed.drop('College_ID', axis=1)
print(f"   Remaining columns: {df_processed.shape[1]}")

# 2. Encode categorical variables
print("\n2. Encoding categorical variables...")
le_internship = LabelEncoder()
le_placement = LabelEncoder()

df_processed['Internship_Experience'] = le_internship.fit_transform(df_processed['Internship_Experience'])
df_processed['Placement'] = le_placement.fit_transform(df_processed['Placement'])

print(f"   Internship_Experience mapping: {dict(zip(le_internship.classes_, le_internship.transform(le_internship.classes_)))}")
print(f"   Placement mapping: {dict(zip(le_placement.classes_, le_placement.transform(le_placement.classes_)))}")

# Display processed data info
print("\n3. Processed Dataset Info:")
print(df_processed.info())
print("\n   First 5 rows of processed data:")
print(df_processed.head())

# Save processed data
df_processed.to_csv('processed_data.csv', index=False)
print("\n✓ Processed data saved to 'processed_data.csv'")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 15))

# 1. Target Variable Distribution
plt.subplot(3, 3, 1)
placement_counts = df['Placement'].value_counts()
colors = ['#ff6b6b', '#51cf66']
plt.bar(placement_counts.index, placement_counts.values, color=colors, edgecolor='black', linewidth=1.5)
plt.title('Placement Distribution (Imbalanced Dataset)', fontsize=14, fontweight='bold')
plt.xlabel('Placement Status', fontsize=12)
plt.ylabel('Count', fontsize=12)
for i, v in enumerate(placement_counts.values):
    plt.text(i, v + 100, f'{v}\n({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# 2. CGPA Distribution by Placement
plt.subplot(3, 3, 2)
df.boxplot(column='CGPA', by='Placement', ax=plt.gca(), patch_artist=True)
plt.title('CGPA Distribution by Placement Status', fontsize=14, fontweight='bold')
plt.xlabel('Placement Status', fontsize=12)
plt.ylabel('CGPA', fontsize=12)
plt.suptitle('')

# 3. IQ Distribution by Placement
plt.subplot(3, 3, 3)
df.boxplot(column='IQ', by='Placement', ax=plt.gca(), patch_artist=True)
plt.title('IQ Distribution by Placement Status', fontsize=14, fontweight='bold')
plt.xlabel('Placement Status', fontsize=12)
plt.ylabel('IQ', fontsize=12)
plt.suptitle('')

# 4. Internship Experience vs Placement
plt.subplot(3, 3, 4)
internship_placement = pd.crosstab(df['Internship_Experience'], df['Placement'])
internship_placement.plot(kind='bar', stacked=False, ax=plt.gca(), color=['#ff6b6b', '#51cf66'], edgecolor='black')
plt.title('Internship Experience vs Placement', fontsize=14, fontweight='bold')
plt.xlabel('Internship Experience', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Placement', labels=['No', 'Yes'])
plt.grid(axis='y', alpha=0.3)

# 5. Projects Completed Distribution
plt.subplot(3, 3, 5)
projects_placement = df.groupby(['Projects_Completed', 'Placement']).size().unstack(fill_value=0)
projects_placement.plot(kind='bar', ax=plt.gca(), color=['#ff6b6b', '#51cf66'], edgecolor='black')
plt.title('Projects Completed vs Placement', fontsize=14, fontweight='bold')
plt.xlabel('Number of Projects', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Placement', labels=['No', 'Yes'])
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)

# 6. Communication Skills Distribution
plt.subplot(3, 3, 6)
comm_placement = df.groupby(['Communication_Skills', 'Placement']).size().unstack(fill_value=0)
comm_placement.plot(kind='bar', ax=plt.gca(), color=['#ff6b6b', '#51cf66'], edgecolor='black')
plt.title('Communication Skills vs Placement', fontsize=14, fontweight='bold')
plt.xlabel('Communication Skills Score', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Placement', labels=['No', 'Yes'])
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)

# 7. Extra Curricular Score Distribution
plt.subplot(3, 3, 7)
df.boxplot(column='Extra_Curricular_Score', by='Placement', ax=plt.gca(), patch_artist=True)
plt.title('Extra Curricular Score by Placement', fontsize=14, fontweight='bold')
plt.xlabel('Placement Status', fontsize=12)
plt.ylabel('Extra Curricular Score', fontsize=12)
plt.suptitle('')

# 8. Academic Performance Distribution
plt.subplot(3, 3, 8)
academic_placement = df.groupby(['Academic_Performance', 'Placement']).size().unstack(fill_value=0)
academic_placement.plot(kind='bar', ax=plt.gca(), color=['#ff6b6b', '#51cf66'], edgecolor='black')
plt.title('Academic Performance vs Placement', fontsize=14, fontweight='bold')
plt.xlabel('Academic Performance Score', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Placement', labels=['No', 'Yes'])
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)

# 9. Previous Semester Result Distribution
plt.subplot(3, 3, 9)
df.boxplot(column='Prev_Sem_Result', by='Placement', ax=plt.gca(), patch_artist=True)
plt.title('Previous Semester Result by Placement', fontsize=14, fontweight='bold')
plt.xlabel('Placement Status', fontsize=12)
plt.ylabel('Previous Semester Result', fontsize=12)
plt.suptitle('')

plt.tight_layout()
plt.savefig('feature_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Feature analysis visualization saved to 'feature_analysis.png'")

# ============================================================================
# CORRELATION HEATMAP
# ============================================================================
fig2, ax = plt.subplots(figsize=(12, 10))
correlation_matrix = df_processed.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            ax=ax)
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Correlation heatmap saved to 'correlation_heatmap.png'")

# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("CORRELATION WITH TARGET VARIABLE (PLACEMENT)")
print("=" * 80)
target_correlation = df_processed.corr()['Placement'].sort_values(ascending=False)
print(target_correlation)

# Visualize feature importance based on correlation
fig3, ax = plt.subplots(figsize=(10, 8))
feature_corr = target_correlation.drop('Placement')
colors_corr = ['#51cf66' if x > 0 else '#ff6b6b' for x in feature_corr.values]
feature_corr.plot(kind='barh', ax=ax, color=colors_corr, edgecolor='black', linewidth=1.5)
plt.title('Feature Correlation with Placement', fontsize=16, fontweight='bold')
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Feature importance visualization saved to 'feature_importance.png'")

print("\n" + "=" * 80)
print("PREPROCESSING & VISUALIZATION COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\nGenerated Files:")
print("1. processed_data.csv - Cleaned and encoded dataset")
print("2. feature_analysis.png - Comprehensive feature analysis")
print("3. correlation_heatmap.png - Feature correlation matrix")
print("4. feature_importance.png - Feature importance based on correlation")
