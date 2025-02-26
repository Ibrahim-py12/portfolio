import streamlit as st
from matplotlib.pyplot import title

aboutme_page = st.Page(

    page = "aboutme.py",
    title = "About me",
    default = True,

)
pr1 = st.Page(
    page = "chatbot.py",
    title = "Automate cursor",

)

pr2 = st.Page(
    page="mathgesai.py",
    title="Math Gesture AI"


)

pg = st.navigation(pages=[aboutme_page,pr1,pr2])

pg.run()





#            streamlit run myapp.py