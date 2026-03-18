import streamlit as st
import sys
import os
from model_recomendation_v2 import generate_response


st.set_page_config(
    page_title="Recomendador de Séries",
    page_icon="🎬"
)

st.title("Qual série assistir hoje?")

st.write(
"""
Esse sistema usa suas séries assistidas no app TV TIME para gerar recomendações personalizadas.
"""
)

query = st.text_input(
    "What do you want to watch?",
    placeholder="ex: sci-fi with complex time travel"
)


if st.button("Gerar recomendação"):

    if query:

        with st.spinner("Pensando..."):

            result = generate_response(query)

        st.subheader("Sugestões")

        st.write(result)