import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load features
with open("features.pkl", "rb") as f:
    data = pickle.load(f)

X = data["features"]
y = data["labels"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 NEW MODEL
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("✅ Model trained successfully")
print("🎯 Accuracy:", accuracy)

# Save model
joblib.dump(model, "model.pkl")