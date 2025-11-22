import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('college_student_placement_dataset.csv')

# Basic information
print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"\nDataset Shape: {df.shape}")
print(f"Number of Rows: {df.shape[0]}")
print(f"Number of Columns: {df.shape[1]}")

print("\n" + "=" * 80)
print("COLUMN NAMES AND DATA TYPES")
print("=" * 80)
print(df.dtypes)

print("\n" + "=" * 80)
print("FIRST 10 ROWS")
print("=" * 80)
print(df.head(10))

print("\n" + "=" * 80)
print("STATISTICAL SUMMARY")
print("=" * 80)
print(df.describe())

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)
print(df.isnull().sum())

print("\n" + "=" * 80)
print("DUPLICATE ROWS")
print("=" * 80)
print(f"Number of Duplicate Rows: {df.duplicated().sum()}")

print("\n" + "=" * 80)
print("TARGET VARIABLE DISTRIBUTION")
print("=" * 80)
if 'Placement' in df.columns:
    print(df['Placement'].value_counts())
    placed_count = (df['Placement'] == 'Yes').sum()
    print(f"\nPlacement Rate: {(placed_count / len(df)) * 100:.2f}%")
else:
    print("Target variable 'Placement' not found. Available columns:")
    print(df.columns.tolist())

print("\n" + "=" * 80)
print("UNIQUE VALUES IN EACH COLUMN")
print("=" * 80)
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")
