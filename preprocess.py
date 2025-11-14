import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
import seaborn as sns
import matplotlib.pyplot as plt

# Load the large dataset
df = pd.read_csv("C:/Users/S.Amrutha/OneDrive/Desktop/2-YEAR/AIML/Datasets/crop_yield.csv")

print("=== LARGE DATASET PREPROCESSING ===")
print(f"Dataset shape: {df.shape}")

# 1. Data Quality Check
print("\n1. DATA QUALITY CHECK")
print("Missing values:", df.isnull().sum().sum())
print("Data types:\n", df.dtypes)

# 2. Feature Engineering
print("\n2. FEATURE ENGINEERING")
df['Growing_Period'] = df['Harvest_Month'] - df['Planting_Month']
df['Growing_Period'] = df['Growing_Period'].apply(lambda x: x + 12 if x < 0 else x)
df['NP_Ratio'] = df['Soil_N_kg_ha'] / df['Soil_P_kg_ha']
df['NK_Ratio'] = df['Soil_N_kg_ha'] / df['Soil_K_kg_ha']
df['PK_Ratio'] = df['Soil_P_kg_ha'] / df['Soil_K_kg_ha']
df['Rainfall_Temp_Ratio'] = df['Avg_Rainfall_mm'] / df['Avg_Temp_C']
df['Soil_Quality_Index'] = (df['Soil_N_kg_ha'] + df['Soil_P_kg_ha'] + df['Soil_K_kg_ha']) / 3

# 3. Categorical Encoding
print("\n3. CATEGORICAL ENCODING")
categorical_cols = ['State', 'District', 'Crop', 'Season']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[f'{col}_Encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

# One-hot encoding for Season
season_dummies = pd.get_dummies(df['Season'], prefix='Season')
df = pd.concat([df, season_dummies], axis=1)

# 4. Outlier Handling
print("\n4. OUTLIER HANDLING")
numerical_cols = ['Soil_pH', 'Soil_N_kg_ha', 'Soil_P_kg_ha', 'Soil_K_kg_ha',
                 'Avg_Rainfall_mm', 'Avg_Temp_C', 'Yield_kg_ha']

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower_bound, upper_bound)

# 5. Feature Scaling
print("\n5. FEATURE SCALING")
features_to_scale = ['Soil_pH', 'Soil_N_kg_ha', 'Soil_P_kg_ha', 'Soil_K_kg_ha',
                    'Avg_Rainfall_mm', 'Avg_Temp_C', 'Planting_Month', 'Harvest_Month',
                    'Growing_Period', 'NP_Ratio', 'NK_Ratio', 'PK_Ratio',
                    'Rainfall_Temp_Ratio', 'Soil_Quality_Index']

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# 6. Feature Selection
print("\n6. FEATURE SELECTION")
X = df_scaled.drop(['State', 'District', 'Crop', 'Season', 'Yield_kg_ha'], axis=1)
X = X.select_dtypes(include=[np.number])  # Keep only numerical columns
y = df_scaled['Yield_kg_ha']

# Select top 15 features
selector = SelectKBest(score_func=f_regression, k=15)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

print("Selected features:", list(selected_features))

# 7. Train-Test Split
print("\n7. TRAIN-TEST SPLIT")
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=df['Crop_Encoded']
)

print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# 8. Save Processed Data
print("\n8. SAVING PROCESSED DATA")
final_features = list(selected_features) + ['Yield_kg_ha']
final_df = df_scaled[final_features]

# Save processed datasets
final_df.to_csv('large_crop_yield_processed.csv', index=False)
pd.DataFrame(X_train).to_csv('X_train_large.csv', index=False)
pd.DataFrame(X_test).to_csv('X_test_large.csv', index=False)
y_train.to_csv('y_train_large.csv', index=False)
y_test.to_csv('y_test_large.csv', index=False)

# Save feature names
feature_names = pd.DataFrame({'feature_names': selected_features})
feature_names.to_csv('feature_names.csv', index=False)

print("\n=== PREPROCESSING COMPLETE ===")
print(f"Final dataset shape: {final_df.shape}")
print("Files saved:")
print("- large_crop_yield_processed.csv")
print("- X_train_large.csv, X_test_large.csv")
print("- y_train_large.csv, y_test_large.csv")
print("- feature_names.csv")

# 9. Dataset Statistics
print("\n9. DATASET STATISTICS")
print(f"Total samples: {len(final_df)}")
print(f"Number of features: {len(final_df.columns) - 1}")
print(f"Crop distribution:\n{df['Crop'].value_counts()}")
print(f"Season distribution:\n{df['Season'].value_counts()}")
print(f"Yield statistics:\nMin: {final_df['Yield_kg_ha'].min():.0f}, "
      f"Max: {final_df['Yield_kg_ha'].max():.0f}, "
      f"Mean: {final_df['Yield_kg_ha'].mean():.0f}")