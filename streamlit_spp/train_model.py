# train_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Generate data
np.random.seed(42)
size = np.random.normal(1500, 500, 100)
price = size * 100 + np.random.normal(0, 10000, 100)
df = pd.DataFrame({'size_sqft': size, 'price': price})

# Train model
model = LinearRegression()
model.fit(df[['size_sqft']], df['price'])

# Save model
joblib.dump(model, 'model.pkl')
