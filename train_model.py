import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load GSI landslide dataset
df = pd.read_csv("gsi_ner_landslides.csv")

print("Dataset loaded successfully!")
print("Total records:", len(df))
print("Columns:", df.columns.tolist())

# Create training features from available location data
X = df[["latitude", "longitude"]].copy()

# Create 4 prototype risk classes for ML training
risk_score = (
    df["latitude"].rank(pct=True) * 0.5 +
    df["longitude"].rank(pct=True) * 0.5
)

df["risk"] = pd.qcut(
    risk_score,
    q=4,
    labels=[1, 2, 3, 4],
    duplicates="drop"
).astype(int)

y = df["risk"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Save trained model
joblib.dump(model, "landslide_model.pkl")

print("SUCCESS: landslide_model.pkl created!")