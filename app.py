import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_precision_recall_curve, classification_report
from utils.clean import clean
from utils.model import make_model, features_importance
from utils.data import load_data

st.title("Bank Churn Project")
st.write("A BeCode project about predicting churning customers")

df = load_data("assets/BankChurners.csv")
st.header("Dataset")
st.write(
    "Our dataset, available on kaggle: https://www.kaggle.com/gilrolnik/"
    "bank-churners")
st.write(df.head(5))

df = clean(df)
x = df.drop("Attrition_Flag", axis=1)
y = df.Attrition_Flag
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state=123)
st.sidebar.title("Model Hyperparameters")
n_est_slider = st.sidebar.slider("n_estimators", 100, 1000, 600, 100)

st.header("Model")

with st.spinner("Training the model..."):
    model = make_model(
        X_train, y_train,
        n_estimators=n_est_slider,
        random_state=123
    )

fig, ax = plt.subplots()
plot_precision_recall_curve(
    model,
    X_test,
    y_test,
    ax=ax,
    pos_label=0,
    name="LightBGM"
)

st.subheader("Precision Recall curve")
st.pyplot(fig)

y_pred = model.predict(X_test)


report = classification_report(y_test, y_pred, output_dict=True)
st.subheader("Classification report")
st.table(pd.DataFrame(report).transpose())
st.subheader("Feature Importance")
st.write(features_importance(model))