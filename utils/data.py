import pandas as pd
import streamlit as st


@st.cache
def load_data(fp):
    return pd.read_csv(fp)