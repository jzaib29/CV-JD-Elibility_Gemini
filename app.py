import streamlit as st
import PyPDF2
import os
from crew_logic import optimize_cv_adversarial

st.set_page_config(page_title="Adversarial CV Optimizer", layout="wide")

st.title("Adversarial CV Optimizer")
st.markdown("Watch an AI Recruiter and AI Resume Writer negotiate to perfect your CV.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API Key", type="password")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        
    target_score = st.slider("Target ATS Score", min_value=5, max_value=10, value=8)
    max_loops = st.slider("Max Iterations", min_value=1, max_value=5, value=3)
    st.markdown("---")
    st.caption("Powered by CrewAI & gpt-oss-120b on Groq")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    
with col2:
    jd_text = st.text_area("Paste Target JD Here", height=150)

if st.button("Start Adversarial Loop", type="primary", use_container_width=True):
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Please enter your Groq API Key in the sidebar.")
    elif uploaded_file and jd_text:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        cv_text = "".join(page.extract_text() for page in pdf_reader.pages)
        
        st.markdown("### Live Agent Logs")
        log_container = st.container(height=300, border=True)
        
        def write_log(message):
            log_container.markdown(message)
            log_container.divider()

        with st.spinner("Agents are negotiating..."):
            result = optimize_cv_adversarial(
                cv_text=cv_text, 
                jd_text=jd_text, 
                target_score=target_score, 
                max_iterations=max_loops,
                log_callback=write_log
            )
            
        st.success(f"Final Score: {result['final_score']}/10 (Completed in {result['iterations_used']} iterations)")
        
        st.markdown("### Optimized CV Output")
        st.text_area("Final CV Text", value=result['updated_cv'], height=400)
        
        st.download_button(
            label="Download Optimized CV",
            data=result['updated_cv'],
            file_name="optimized_cv.txt",
            mime="text/plain"
        )
    else:
        st.warning("Provide both the CV and Job Description.")
