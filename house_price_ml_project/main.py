import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1. Create output folder
# -------------------------------
os.makedirs("output", exist_ok=True)

# -------------------------------
# 2. Load data
# -------------------------------
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Save test IDs
test_ids = test["Id"]

# Drop Id
train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)

# -------------------------------
# 3. Separate target
# -------------------------------
y = train["SalePrice"]
train.drop("SalePrice", axis=1, inplace=True)

# -------------------------------
# 4. Combine for preprocessing
# -------------------------------
full = pd.concat([train, test], axis=0)

# -------------------------------
# 5. Handle missing values (FIXED)
# -------------------------------
# Categorical columns
cat_cols = full.select_dtypes(include=["object"]).columns

# Numerical columns
num_cols = full.select_dtypes(exclude=["object"]).columns

# Fill categorical
for col in cat_cols:
    full[col] = full[col].fillna("None")

# Fill numerical
for col in num_cols:
    full[col] = full[col].fillna(full[col].median())

# -------------------------------
# 6. Convert categorical → numeric
# -------------------------------
full = pd.get_dummies(full)

# -------------------------------
# 7. Split back
# -------------------------------
X = full[:len(train)]
X_test_final = full[len(train):]

# -------------------------------
# 8. Train-test split
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 9. Model (Advanced)
# -------------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------------
# 10. Evaluation
# -------------------------------
y_pred = model.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("🔥 Advanced Model Performance:")
print("MSE:", mse)
print("R2 Score:", r2)

# -------------------------------
# 11. Plot
# -------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_val, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted (Advanced Model)")
plt.savefig("output/advanced_plot.png")

# -------------------------------
# 12. Final Predictions
# -------------------------------
final_preds = model.predict(X_test_final)

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": final_preds
})

submission.to_csv("output/advanced_submission.csv", index=False)

print("✅ Advanced submission file created!")