import streamlit as st
import matplotlib.pyplot as plt
from plot import create_chart

st.title('My Visualization')

chart = create_chart()
st.altair_chart(chart, use_container_width=True)