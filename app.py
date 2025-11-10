"""
AIsthma Forge - Microbiome Analysis Platform for Asthma Research
A comprehensive web application for analyzing microbiome sequencing data
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AIsthma Forge",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🫁 AIsthma Forge</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Transform Microbiome Data into Actionable Asthma Insights</p>', unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'otu_table' not in st.session_state:
    st.session_state.otu_table = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'functional_data' not in st.session_state:
    st.session_state.functional_data = None

# Sidebar navigation
st.sidebar.title("📊 Analysis Pipeline")
analysis_step = st.sidebar.radio(
    "Select Analysis Step:",
    [
        "🏠 Home & Data Upload",
        "🔧 Preprocessing",
        "📈 Diversity Analysis",
        "🧬 Differential Abundance",
        "🤖 Machine Learning",
        "🔬 Predictive Modeling",
        "📄 Reports & Export"
    ]
)

# Main content based on selection
if analysis_step == "🏠 Home & Data Upload":
    from modules.data_upload import render_upload_page
    render_upload_page()

elif analysis_step == "🔧 Preprocessing":
    from modules.preprocessing import render_preprocessing_page
    render_preprocessing_page()

elif analysis_step == "📈 Diversity Analysis":
    from modules.diversity import render_diversity_page
    render_diversity_page()

elif analysis_step == "🧬 Differential Abundance":
    from modules.differential_abundance import render_differential_page
    render_differential_page()

elif analysis_step == "🤖 Machine Learning":
    from modules.machine_learning import render_ml_page
    render_ml_page()

elif analysis_step == "🔬 Predictive Modeling":
    from modules.predictive_modeling import render_predictive_page
    render_predictive_page()

elif analysis_step == "📄 Reports & Export":
    from modules.reports import render_reports_page
    render_reports_page()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 About")
st.sidebar.info("""
**AIsthma Forge** is an open-source platform for microbiome analysis in asthma research.

**Features:**
- FASTQ/OTU/ASV data processing
- Diversity metrics & visualization
- Differential abundance (pydeseq2)
- ML classification (RF/XGBoost + SHAP)
- Predictive gene modeling
- Clinical risk scoring

**Version:** 1.0.0
""")
