import streamlit as st

st.set_page_config(
    page_title="Quantitative Finance Group USM",
    page_icon="👋",
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

st.write("# Welcome to Quantitative Finance Group USM! 👋")

