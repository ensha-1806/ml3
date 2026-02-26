# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Page Title
st.title("🌸 KNN Classification App with Plotly")

# Load Dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Sidebar for K value
k = st.sidebar.slider("Select K value", 1, 15, 5)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# KNN Model
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.subheader("Model Accuracy")
st.success(f"Accuracy: {accuracy * 100:.2f}%")

# Plotly Scatter Plot
st.subheader("Interactive Scatter Plot")

fig = px.scatter(
    X,
    x="sepal length (cm)",
    y="sepal width (cm)",
    color=y.astype(str),
    title="Sepal Length vs Sepal Width",
    labels={"color": "Species"}
)

st.plotly_chart(fig)
