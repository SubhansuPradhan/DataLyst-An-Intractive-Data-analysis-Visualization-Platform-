import streamlit as st

def load_header():
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image("https://i.postimg.cc/K89BXfnm/logo1.gif", width=130)
    with col2:
        st.markdown("""
            <h1 style='
                font-size: 90px;
                font-weight: bold;
                background: linear-gradient(90deg, #6a00f4, #00d084, #ff6900);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: left;
            '>DataLyst</h1>
        """, unsafe_allow_html=True)
