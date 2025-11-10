"""
Data Upload Module
Handles upload of FASTQ, OTU/ASV counts, metadata, and functional predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import os

def render_upload_page():
    """Render the data upload page"""
    
    st.header("📁 Data Upload & Configuration")
    
    st.markdown("""
    Welcome to **AIsthma Forge**! This platform enables comprehensive microbiome analysis 
    for asthma research. Upload your data files to begin the analysis pipeline.
    
    ### Supported Data Types:
    - **OTU/ASV Count Table**: Tab-separated or CSV file with taxa as rows, samples as columns
    - **Metadata**: Sample information including asthma status, age, treatment groups, etc.
    - **Functional Predictions** (optional): PICRUSt2, HUMAnN, or other functional pathway data
    """)
    
    st.markdown("---")
    
    # Create tabs for different upload methods
    tab1, tab2, tab3 = st.tabs(["📤 Upload Files", "🔗 Load Example Data", "📋 Manual Entry"])
    
    with tab1:
        st.subheader("Upload Your Data Files")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### OTU/ASV Count Table")
            otu_file = st.file_uploader(
                "Upload count table (CSV/TSV)",
                type=['csv', 'tsv', 'txt'],
                key='otu_upload',
                help="Rows = taxa/features, Columns = samples"
            )
            
            if otu_file is not None:
                try:
                    # Try to read with different delimiters
                    content = otu_file.getvalue().decode('utf-8')
                    if '\t' in content:
                        otu_df = pd.read_csv(StringIO(content), sep='\t', index_col=0)
                    else:
                        otu_df = pd.read_csv(StringIO(content), index_col=0)
                    
                    st.success(f"✅ Loaded OTU table: {otu_df.shape[0]} features × {otu_df.shape[1]} samples")
                    st.session_state.otu_table = otu_df
                    
                    with st.expander("Preview OTU Table"):
                        st.dataframe(otu_df.head(10))
                        st.write(f"**Data type:** {otu_df.values.dtype}")
                        st.write(f"**Total counts:** {otu_df.sum().sum():,.0f}")
                        
                except Exception as e:
                    st.error(f"Error reading OTU file: {str(e)}")
        
        with col2:
            st.markdown("#### Metadata")
            metadata_file = st.file_uploader(
                "Upload metadata (CSV/TSV)",
                type=['csv', 'tsv', 'txt'],
                key='metadata_upload',
                help="Must include 'SampleID' and 'Asthma_Status' columns"
            )
            
            if metadata_file is not None:
                try:
                    content = metadata_file.getvalue().decode('utf-8')
                    if '\t' in content:
                        metadata_df = pd.read_csv(StringIO(content), sep='\t')
                    else:
                        metadata_df = pd.read_csv(StringIO(content))
                    
                    # Check for required columns
                    if 'SampleID' not in metadata_df.columns:
                        if metadata_df.index.name == 'SampleID' or metadata_df.index.name is None:
                            metadata_df = metadata_df.reset_index()
                            if 'index' in metadata_df.columns:
                                metadata_df = metadata_df.rename(columns={'index': 'SampleID'})
                    
                    st.success(f"✅ Loaded metadata: {len(metadata_df)} samples")
                    st.session_state.metadata = metadata_df
                    
                    with st.expander("Preview Metadata"):
                        st.dataframe(metadata_df.head(10))
                        st.write(f"**Columns:** {', '.join(metadata_df.columns)}")
                        
                        # Show distribution of key variables
                        if 'Asthma_Status' in metadata_df.columns:
                            st.write("**Asthma Status Distribution:**")
                            st.write(metadata_df['Asthma_Status'].value_counts())
                        
                except Exception as e:
                    st.error(f"Error reading metadata file: {str(e)}")
        
        # Functional data upload
        st.markdown("---")
        st.markdown("#### Functional Predictions (Optional)")
        
        functional_file = st.file_uploader(
            "Upload functional data (PICRUSt2/HUMAnN output)",
            type=['csv', 'tsv', 'txt'],
            key='functional_upload',
            help="Gene families, pathways, or KO abundances"
        )
        
        if functional_file is not None:
            try:
                content = functional_file.getvalue().decode('utf-8')
                if '\t' in content:
                    func_df = pd.read_csv(StringIO(content), sep='\t', index_col=0)
                else:
                    func_df = pd.read_csv(StringIO(content), index_col=0)
                
                st.success(f"✅ Loaded functional data: {func_df.shape[0]} features × {func_df.shape[1]} samples")
                st.session_state.functional_data = func_df
                
                with st.expander("Preview Functional Data"):
                    st.dataframe(func_df.head(10))
                    
            except Exception as e:
                st.error(f"Error reading functional file: {str(e)}")
    
    with tab2:
        st.subheader("Load Example Datasets")
        
        st.markdown("""
        Load pre-configured example datasets to explore the platform's capabilities:
        """)
        
        example_dataset = st.selectbox(
            "Select Example Dataset:",
            [
                "None",
                "TeoSM_2015 - Nasal Microbiome (Asthma vs Healthy)",
                "COPSAC - Gut Microbiome (Early-life Asthma Risk)",
                "Synthetic Dataset - Demo Analysis"
            ]
        )
        
        if st.button("Load Example Dataset"):
            if example_dataset == "Synthetic Dataset - Demo Analysis":
                # Generate synthetic data for demonstration
                np.random.seed(42)
                
                # Create synthetic OTU table
                n_samples = 60
                n_features = 50
                
                sample_names = [f"Sample_{i+1}" for i in range(n_samples)]
                feature_names = [
                    f"Bacteria|Firmicutes|Clostridia|Clostridiales|Lachnospiraceae|{i}" 
                    if i < 20 else f"Bacteria|Bacteroidetes|Bacteroidia|Bacteroidales|Bacteroidaceae|{i}"
                    if i < 35 else f"Bacteria|Proteobacteria|Gammaproteobacteria|Enterobacterales|Enterobacteriaceae|{i}"
                    for i in range(n_features)
                ]
                
                # Generate counts with some structure
                otu_data = np.random.negative_binomial(5, 0.3, size=(n_features, n_samples))
                otu_df = pd.DataFrame(otu_data, index=feature_names, columns=sample_names)
                
                # Create synthetic metadata
                metadata_df = pd.DataFrame({
                    'SampleID': sample_names,
                    'Asthma_Status': ['Asthma'] * 30 + ['Healthy'] * 30,
                    'Age': np.random.randint(5, 18, n_samples),
                    'Sex': np.random.choice(['Male', 'Female'], n_samples),
                    'BMI': np.random.normal(18, 3, n_samples).round(1),
                    'Site': np.random.choice(['Nasal', 'Gut'], n_samples)
                })
                
                st.session_state.otu_table = otu_df
                st.session_state.metadata = metadata_df
                st.session_state.data_loaded = True
                
                st.success("✅ Synthetic dataset loaded successfully!")
                st.info(f"Loaded {n_features} features across {n_samples} samples (30 Asthma, 30 Healthy)")
                
            else:
                st.warning("Example datasets from public repositories are not yet implemented. Please use the Synthetic Dataset for demonstration.")
    
    with tab3:
        st.subheader("Manual Data Entry")
        st.markdown("For small datasets, you can manually enter data:")
        
        st.text_area(
            "Paste OTU table (tab-separated)",
            height=200,
            help="First row: sample names, First column: feature IDs"
        )
        
        st.text_area(
            "Paste metadata (tab-separated)",
            height=200,
            help="First row: column names, First column: sample IDs"
        )
        
        if st.button("Parse Manual Entry"):
            st.warning("Manual entry parsing not yet implemented. Please use file upload.")
    
    # Data validation and summary
    st.markdown("---")
    st.subheader("📊 Data Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.otu_table is not None:
            st.metric("OTU/ASV Features", st.session_state.otu_table.shape[0])
            st.metric("Samples", st.session_state.otu_table.shape[1])
        else:
            st.info("No OTU data loaded")
    
    with col2:
        if st.session_state.metadata is not None:
            st.metric("Metadata Samples", len(st.session_state.metadata))
            if 'Asthma_Status' in st.session_state.metadata.columns:
                asthma_counts = st.session_state.metadata['Asthma_Status'].value_counts()
                st.write("**Groups:**")
                for group, count in asthma_counts.items():
                    st.write(f"- {group}: {count}")
        else:
            st.info("No metadata loaded")
    
    with col3:
        if st.session_state.functional_data is not None:
            st.metric("Functional Features", st.session_state.functional_data.shape[0])
            st.success("Functional data available")
        else:
            st.info("No functional data loaded")
    
    # Validation checks
    if st.session_state.otu_table is not None and st.session_state.metadata is not None:
        st.markdown("---")
        st.subheader("✅ Data Validation")
        
        otu_samples = set(st.session_state.otu_table.columns)
        meta_samples = set(st.session_state.metadata['SampleID']) if 'SampleID' in st.session_state.metadata.columns else set()
        
        common_samples = otu_samples & meta_samples
        
        if len(common_samples) > 0:
            st.success(f"✅ {len(common_samples)} samples match between OTU table and metadata")
            st.session_state.data_loaded = True
            
            if len(common_samples) < len(otu_samples):
                st.warning(f"⚠️ {len(otu_samples) - len(common_samples)} samples in OTU table not found in metadata")
            if len(common_samples) < len(meta_samples):
                st.warning(f"⚠️ {len(meta_samples) - len(common_samples)} samples in metadata not found in OTU table")
            
            st.info("✨ Data is ready for analysis! Proceed to **Preprocessing** in the sidebar.")
        else:
            st.error("❌ No matching samples found between OTU table and metadata. Please check sample IDs.")
            st.session_state.data_loaded = False
