import streamlit as st
import pandas as pd

# Backend functions (will be created in Part 2)
from ats import (
    extract_text,
    calculate_ats_score,
    extract_skills,
    missing_skills
)

st.set_page_config(
    page_title="AI Resume Screening ATS",
    page_icon="📄",
    layout="wide"
)

# -----------------------
# Header
# -----------------------

st.title("📄 AI Resume Screening ATS")
st.markdown("---")

st.write("""
Upload your Resume and Job Description.

The ATS will

✅ Calculate ATS Match Score

✅ Extract Skills

✅ Identify Missing Skills

✅ Recommend Improvements

✅ Predict Resume Category (optional)
""")

# -----------------------
# Upload Files
# -----------------------

col1, col2 = st.columns(2)

with col1:

    resume = st.file_uploader(
        "Upload Resume",
        type=["pdf", "docx"]
    )

with col2:

    job_description = st.file_uploader(
        "Upload Job Description",
        type=["pdf", "docx", "txt"]
    )

st.markdown("---")

# -----------------------
# Analyze Button
# -----------------------

if st.button("Analyze Resume"):

    if resume is None or job_description is None:

        st.error("Please upload both Resume and Job Description.")

    else:

        with st.spinner("Analyzing Resume..."):

            resume_text = extract_text(resume)

            jd_text = extract_text(job_description)

            score = calculate_ats_score(
                resume_text,
                jd_text
            )

            resume_skills = extract_skills(
                resume_text
            )

            jd_skills = extract_skills(
                jd_text
            )

            missing = missing_skills(
                resume_skills,
                jd_skills
            )

        st.success("Analysis Completed")

        st.markdown("---")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "ATS Score",
                f"{score:.2f}%"
            )

        with c2:

            st.metric(
                "Resume Skills",
                len(resume_skills)
            )

        with c3:

            st.metric(
                "Missing Skills",
                len(missing)
            )

        st.markdown("---")

        left, right = st.columns(2)

        with left:

            st.subheader("Resume Skills")

            st.write(resume_skills)

        with right:

            st.subheader("Job Description Skills")

            st.write(jd_skills)

        st.markdown("---")

        st.subheader("Missing Skills")

        if len(missing) == 0:

            st.success("Excellent! No missing skills found.")

        else:

            st.warning(missing)

        st.markdown("---")

        st.subheader("Resume Preview")

        st.text_area(
            "Resume",
            resume_text,
            height=250
        )

        st.subheader("Job Description Preview")

        st.text_area(
            "Job Description",
            jd_text,
            height=250
        )
