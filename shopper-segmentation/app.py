import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models and data
kmeans = joblib.load("kmeans_rfm_model.pkl")
scaler = joblib.load("rfm_scaler.pkl")
product_similarity = joblib.load("product_similarity.pkl")  # DataFrame
product_list = joblib.load("product_list.pkl")  # List of product names

# Mapping cluster labels
cluster_label_map = {
    0: 'High-Value',
    1: 'At-Risk',
    2: 'Regular',
    3: 'Occasional'
}

# App title
st.title("Shopper Spectrum: Customer Segmentation & Product Recommendation")

# Navigation
app_mode = st.sidebar.selectbox("Choose App Section", ["Customer Segmentation", "Product Recommendation"])

# Section 1: Customer Segmentation
if app_mode == "Customer Segmentation":
    st.subheader("Predict Customer Segment Using RFM")

    # User inputs
    recency = st.number_input("Recency (days since last purchase)", min_value=0, step=1)
    frequency = st.number_input("Frequency (total number of purchases)", min_value=0, step=1)
    monetary = st.number_input("Monetary (total amount spent)", min_value=0.0, step=0.1)

    if st.button("Predict Cluster"):
        input_data = np.array([[recency, frequency, monetary]])
        input_scaled = scaler.transform(input_data)
        cluster = kmeans.predict(input_scaled)[0]
        segment = cluster_label_map.get(cluster, f"Cluster {cluster}")
        st.success(f"This customer belongs to the **{segment}** segment.")

# Section 2: Product Recommendation
elif app_mode == "Product Recommendation":
    st.subheader("Find Similar Products")

    product_name = st.text_input("Enter a product name (case-sensitive)")

    if st.button("Get Recommendations"):
        if product_name not in product_similarity.index:
            st.error("Product not found in database. Please check the spelling or try a different product.")
        else:
            # Get similarity vector for the given product
            similarity_vector = product_similarity.loc[product_name]

            # Get top 5 similar products (excluding the product itself)
            similar_products = similarity_vector.sort_values(ascending=False)[1:6]

            st.write("### Recommended Products:")
            for i, item in enumerate(similar_products.index, 1):
                st.write(f"{i}. {item}")
