import streamlit as st
from streamlit_lottie import st_lottie
import base64

st.set_page_config(
     page_title="NLP Project - Lukas and Stephan",
     page_icon="🐝",
     layout="wide",
)

def main():
    st.title("Uncovering the Authors")
    st.subheader("A Dual Approach to Detect and Understand Authorship in Text Documents")
    st.write("Navigate in the Sidebar through our different Pages to get some interessting insights in some of our analytics we have done for fingerprint analysis and authorship change detection.")
    st_lottie("https://assets9.lottiefiles.com/packages/lf20_bUYZYN2sM0.json", height=200)
    
    with open("NLP_Report.pdf", "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.subheader("Project Report")
    st.markdown(pdf_display, unsafe_allow_html=True)
    
    
if __name__ == '__main__':
    main()
