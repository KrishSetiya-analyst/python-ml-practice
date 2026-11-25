import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Load trained model
model = joblib.load("model.pkl")

# Generate synthetic data (for visualization)
def generate_data():
    np.random.seed(42)
    size = np.random.normal(1500, 500, 100)
    price = size * 100 + np.random.normal(0, 10000, 100)
    return pd.DataFrame({'size_sqft': size, 'price': price})

# Streamlit App
def main():
    st.title("🏡 House Price Predictor App")
    st.markdown("Enter house size (in sqft) and predict the price.")

    # Input
    size = st.slider("Select House Size (sqft)", 500, 5000, 1500, step=50)

    if st.button("Predict"):
        prediction = model.predict([[size]])[0]
        st.success(f"💰 Estimated House Price: ${prediction:,.2f}")

        # Show visualization
        df = generate_data()
        fig = px.scatter(df, x='size_sqft', y='price',
                         title="House Size vs Price",
                         labels={'size_sqft': 'Size (sqft)', 'price': 'Price ($)'})
        fig.add_scatter(x=[size], y=[prediction],
                        mode='markers', marker=dict(color='red', size=12),
                        name='Your Prediction')
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
