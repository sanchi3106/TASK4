# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load Dataset
# Replace with your dataset path or use pd.read_csv("your_dataset.csv")
df = pd.read_csv(r"C:\Users\khanc\Downloads\advertising.csv")
# Step 3: Data Exploration
print(df.head())
print("\nDataset Info:\n", df.info())
print("\nMissing Values:\n", df.isnull().sum())

# Step 4: Visualize Correlation
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# Step 5: Define Features and Target
X = df[['TV', 'Radio', 'Newspaper']]  # Features (ad spend)
y = df['Sales']                      # Target (sales)

# Step 6: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Make Predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate Model
print("\nModel Performance:")
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Step 10: Plot Actual vs Predicted Sales
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()
