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

# Create simple training features from available location data
X = df[["latitude", "longitude"]].copy()

# Prototype risk labels based on latitude/longitude distribution
# NOTE: These are demonstration labels, not official GSI risk classes.
df["risk"] = (
    (df["latitude"] * 0.7 + df["longitude"] * 0.3)
    > (df["latitude"] * 0.7 + df["longitude"] * 0.3).median()
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