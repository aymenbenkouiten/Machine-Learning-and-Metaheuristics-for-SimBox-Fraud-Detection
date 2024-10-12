import streamlit as st

st.set_page_config(page_title="Home", page_icon="🏠", layout='centered')


# Titre et en-tête
st.markdown("<h5 style='text-align: center;'>Université des Sciences et de la Technologie Houari Boumediene</h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Faculté d’Informatique</h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Département IA et SD</h5>", unsafe_allow_html=True)

st.markdown(
    """
    <div style='display: flex; justify-content: center;'>
        <img src='https://www.usthb.dz/storage/logos/usthb.svg' width='150'/>
    </div><br>
    """, 
    unsafe_allow_html=True
)

st.markdown("<h5 style='text-align: center;'>Domaine : Informatique</h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Spécialité : Systèmes Informatiques Intelligents</h5>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center;'>Apprentissage automatique pour l’amélioration de la détection SIMBOX dans les télécommunications</h4>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Organisme d’accueil : Djezzy OTA</h5>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("<h5 style='text-align: left;'>Sujet proposé et dirigé par :</h5>", unsafe_allow_html=True)
    st.write("- Mr. BENATMANE Mohamed Amine")
    st.write("- Mr. KHENNAK Ilyes")
with col2:
    st.markdown("<h5 style='text-align: left;'>Présenté par :</h5>", unsafe_allow_html=True)
    st.write("- BENKOUITEN Aymen")
    st.write("- NOUGHI Tarek")
col1, col2, col3 = st.columns(3)
with col2:
    st.markdown("<h5 style='text-align: left;'>Devant :</h5>", unsafe_allow_html=True)
    st.markdown("- Mme. ALIOUANE LYNDA")
    st.markdown("- Mme. SEBAI MERIEM")

st.markdown("<h5 style='text-align: center;'>Binôme no : SII-26 /2024</h5>", unsafe_allow_html=True)
