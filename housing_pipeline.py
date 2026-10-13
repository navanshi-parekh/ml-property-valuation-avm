import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure output directories exist
os.makedirs("plots", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ---------------------------------------------------------
# PHASE 1 & 2: DATA ACQUISITION & EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------
print("--> Loading California Housing dataset...")
raw_data = fetch_california_housing(as_frame=True)
df = raw_data.frame

# Target variable represents median house value in $100,000s
# Multiply by 100,000 to convert to realistic appraisal dollar amounts
df["Price"] = df["MedHouseVal"] * 100000.0
df = df.drop(columns=["MedHouseVal"])

# Save local copy
df.to_csv("data/housing_data.csv", index=False)
print(f"Dataset ready. Observations: {len(df)}, Features: {df.shape[1] - 1}")

# Plot 1: Target Price Distribution
plt.figure(figsize=(7, 4))
sns.histplot(df["Price"], kde=True, color="#4f46e5", bins=40)
plt.title("Distribution of Residential Property Appraisal Values ($)")
plt.xlabel("Property Value ($)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plots/1_price_distribution.png")
plt.close()

# Plot 2: Median Income vs Price Scatter Plot
plt.figure(figsize=(7, 4))
sns.scatterplot(x="MedInc", y="Price", data=df.sample(2000, random_state=42), alpha=0.5, color="#0ea5e9")
plt.title("Locality Median Income vs. Property Price")
plt.xlabel("Median Income (Tens of Thousands $)")
plt.ylabel("Property Price ($)")
plt.tight_layout()
plt.savefig("plots/2_income_vs_price.png")
plt.close()

# Plot 3: Feature Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Correlation Matrix of Housing Attributes")
plt.tight_layout()
plt.savefig("plots/3_correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------
# PHASE 3: PREPROCESSING & FEATURE SELECTION
# ---------------------------------------------------------
print("--> Preprocessing and Scaling features...")

X = df.drop(columns=["Price"])
y = df["Price"]

# Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Technical Justification: StandardScaler ensures distance/regularization-sensitive
# models (Ridge) treat high-magnitude features (Population, AveOccup) equally with low ones (MedInc).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# PHASE 4: MODEL DEVELOPMENT & COMPARISON
# ---------------------------------------------------------
models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    "XGBoost Regressor": XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42, n_jobs=-1)
}

results = {}
predictions = {}

print("--> Training and evaluating regression models...")
for name, model in models.items():
    # Linear models require scaled inputs; tree models can train on either
    train_x = X_train_scaled if "Ridge" in name else X_train
    test_x = X_test_scaled if "Ridge" in name else X_test
    
    model.fit(train_x, y_train)
    y_pred = model.predict(test_x)
    predictions[name] = y_pred
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    results[name] = {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape}

# Plot 4: Actual vs Predicted Valuation Scatter (XGBoost)
best_preds = predictions["XGBoost Regressor"]
plt.figure(figsize=(6, 6))
plt.scatter(y_test.values[:1000], best_preds[:1000], alpha=0.4, color="#10b981", edgecolors="none")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r", linewidth=2, label="Ideal Fit (y = x)")
plt.title("Actual vs. Predicted Property Valuations (XGBoost)")
plt.xlabel("Actual Valuation ($)")
plt.ylabel("Model Predicted Valuation ($)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/4_actual_vs_predicted.png")
plt.close()

# Print metrics to console
print("\n" + "="*55)
print("             REGRESSION EVALUATION BENCHMARK          ")
print("="*55)
for name, m in results.items():
    print(f"[{name}]")
    print(f"  R² Score:  {m['R2']:.4f} (Variance Explained)")
    print(f"  MAE:       ${m['MAE']:,.2f} (Average Dollar Deviation)")
    print(f"  RMSE:      ${m['RMSE']:,.2f}")
    print(f"  MAPE:      {m['MAPE']:.2f}%\n")

# ---------------------------------------------------------
# PHASE 5: MORTGAGE UNDERWRITING & APPRAISAL INTERPRETATION
# ---------------------------------------------------------
print("="*55)
print("     MORTGAGE UNDERWRITING & ENGINEERING IMPACT       ")
print("="*55)
# A valuation within 10% of true value is typically acceptable for automated loan-to-value (LTV) decisions
within_10_pct = np.mean(np.abs((y_test - best_preds) / y_test) <= 0.10) * 100
print(f"Proportion of properties predicted within +/-10% LTV tolerance: {within_10_pct:.2f}%")
print(f"Manual on-site appraisal cost avoided per 1,000 automated evaluations: ${(within_10_pct / 100 * 1000) * 450:,.2f} (at $450/survey)")