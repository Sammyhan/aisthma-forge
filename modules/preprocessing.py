"""
Preprocessing Module
Quality control, filtering, normalization, and transformation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

def render_preprocessing_page():
    """Render the preprocessing page"""
    
    st.header("🔧 Data Preprocessing & Quality Control")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data in the 'Home & Data Upload' section first.")
        return
    
    otu_table = st.session_state.otu_table.copy()
    metadata = st.session_state.metadata.copy()
    
    st.markdown("""
    Preprocessing steps ensure data quality and prepare it for downstream analysis.
    Apply filtering, normalization, and transformation based on your research needs.
    """)
    
    # Create tabs for different preprocessing steps
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Quality Control",
        "🧹 Filtering",
        "📊 Normalization",
        "🔄 Transformation"
    ])
    
    with tab1:
        st.subheader("Quality Control Metrics")
        
        # Calculate QC metrics
        sample_depths = otu_table.sum(axis=0)
        feature_prevalence = (otu_table > 0).sum(axis=1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Sample Sequencing Depth")
            
            # Plot sequencing depth distribution
            fig = px.histogram(
                x=sample_depths.values,
                nbins=30,
                title="Distribution of Sequencing Depth",
                labels={'x': 'Total Reads per Sample', 'y': 'Count'}
            )
            fig.add_vline(
                x=sample_depths.median(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Median: {sample_depths.median():.0f}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"**Mean depth:** {sample_depths.mean():.0f}")
            st.write(f"**Median depth:** {sample_depths.median():.0f}")
            st.write(f"**Min depth:** {sample_depths.min():.0f}")
            st.write(f"**Max depth:** {sample_depths.max():.0f}")
        
        with col2:
            st.markdown("#### Feature Prevalence")
            
            # Plot feature prevalence
            fig = px.histogram(
                x=feature_prevalence.values,
                nbins=30,
                title="Distribution of Feature Prevalence",
                labels={'x': 'Number of Samples', 'y': 'Number of Features'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"**Total features:** {len(feature_prevalence)}")
            st.write(f"**Features in >50% samples:** {(feature_prevalence > len(otu_table.columns)/2).sum()}")
            st.write(f"**Features in >10% samples:** {(feature_prevalence > len(otu_table.columns)*0.1).sum()}")
            st.write(f"**Singleton features:** {(feature_prevalence == 1).sum()}")
        
        # Rarefaction curve
        st.markdown("---")
        st.markdown("#### Rarefaction Curves")
        
        # Sample a subset for visualization
        n_samples_to_plot = min(20, len(otu_table.columns))
        samples_to_plot = np.random.choice(otu_table.columns, n_samples_to_plot, replace=False)
        
        rarefaction_data = []
        depths = np.linspace(0, sample_depths.max(), 20).astype(int)
        
        for sample in samples_to_plot:
            sample_data = otu_table[sample].values
            sample_data = sample_data[sample_data > 0]
            
            for depth in depths:
                if depth == 0:
                    richness = 0
                elif depth >= len(sample_data):
                    richness = len(sample_data)
                else:
                    # Simple approximation
                    richness = len(np.random.choice(sample_data, depth, replace=True))
                
                rarefaction_data.append({
                    'Sample': sample,
                    'Depth': depth,
                    'Richness': richness
                })
        
        rare_df = pd.DataFrame(rarefaction_data)
        fig = px.line(
            rare_df,
            x='Depth',
            y='Richness',
            color='Sample',
            title="Rarefaction Curves (Sample Subset)",
            labels={'Depth': 'Sequencing Depth', 'Richness': 'Observed Features'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Filtering Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Sample Filtering")
            
            min_depth = st.number_input(
                "Minimum sequencing depth",
                min_value=0,
                max_value=int(sample_depths.max()),
                value=int(sample_depths.quantile(0.1)),
                step=100,
                help="Remove samples with fewer reads than this threshold"
            )
            
            samples_passing = (sample_depths >= min_depth).sum()
            st.info(f"✅ {samples_passing} / {len(sample_depths)} samples will be retained")
        
        with col2:
            st.markdown("#### Feature Filtering")
            
            min_prevalence = st.slider(
                "Minimum prevalence (%)",
                min_value=0,
                max_value=100,
                value=10,
                step=5,
                help="Keep features present in at least this % of samples"
            )
            
            min_abundance = st.number_input(
                "Minimum total abundance",
                min_value=0,
                max_value=int(otu_table.sum(axis=1).max()),
                value=10,
                step=10,
                help="Keep features with at least this total count across all samples"
            )
            
            prevalence_threshold = len(otu_table.columns) * (min_prevalence / 100)
            features_passing = ((otu_table > 0).sum(axis=1) >= prevalence_threshold) & \
                              (otu_table.sum(axis=1) >= min_abundance)
            
            st.info(f"✅ {features_passing.sum()} / {len(otu_table)} features will be retained")
        
        st.markdown("---")
        
        if st.button("Apply Filtering", type="primary"):
            # Filter samples
            samples_to_keep = sample_depths >= min_depth
            otu_filtered = otu_table.loc[:, samples_to_keep]
            
            # Filter features
            prevalence_threshold = len(otu_filtered.columns) * (min_prevalence / 100)
            features_to_keep = ((otu_filtered > 0).sum(axis=1) >= prevalence_threshold) & \
                              (otu_filtered.sum(axis=1) >= min_abundance)
            otu_filtered = otu_filtered.loc[features_to_keep, :]
            
            # Update session state
            st.session_state.otu_table_filtered = otu_filtered
            
            # Filter metadata to match
            if 'SampleID' in metadata.columns:
                metadata_filtered = metadata[metadata['SampleID'].isin(otu_filtered.columns)]
            else:
                metadata_filtered = metadata
            st.session_state.metadata_filtered = metadata_filtered
            
            st.success(f"""
            ✅ Filtering complete!
            - Samples: {len(otu_table.columns)} → {len(otu_filtered.columns)}
            - Features: {len(otu_table)} → {len(otu_filtered)}
            """)
    
    with tab3:
        st.subheader("Normalization Methods")
        
        st.markdown("""
        Normalization accounts for differences in sequencing depth and library size.
        Choose the method appropriate for your downstream analysis.
        """)
        
        normalization_method = st.selectbox(
            "Select Normalization Method:",
            [
                "Total Sum Scaling (TSS) / Relative Abundance",
                "Rarefaction (Subsampling)",
                "CSS (Cumulative Sum Scaling)",
                "TMM (Trimmed Mean of M-values)",
                "CLR (Centered Log-Ratio)",
                "None (Keep Raw Counts)"
            ]
        )
        
        # Get the filtered table if available, otherwise use original
        if 'otu_table_filtered' in st.session_state:
            otu_to_normalize = st.session_state.otu_table_filtered.copy()
        else:
            otu_to_normalize = otu_table.copy()
        
        if normalization_method == "Total Sum Scaling (TSS) / Relative Abundance":
            st.info("""
            **TSS** divides each sample by its total count, converting to relative abundances (0-1).
            Best for: Compositional analyses, diversity metrics
            """)
            
            if st.button("Apply TSS Normalization", type="primary"):
                otu_normalized = otu_to_normalize.div(otu_to_normalize.sum(axis=0), axis=1)
                st.session_state.otu_table_normalized = otu_normalized
                st.success("✅ TSS normalization applied!")
                
                with st.expander("Preview Normalized Data"):
                    st.dataframe(otu_normalized.head())
                    st.write(f"Sample sums (should all be 1.0): {otu_normalized.sum(axis=0).head()}")
        
        elif normalization_method == "Rarefaction (Subsampling)":
            min_depth = int(otu_to_normalize.sum(axis=0).min())
            rarefaction_depth = st.number_input(
                "Rarefaction depth",
                min_value=100,
                max_value=min_depth,
                value=min_depth,
                help="All samples will be subsampled to this depth"
            )
            
            st.info(f"""
            **Rarefaction** randomly subsamples each sample to {rarefaction_depth} reads.
            Best for: Alpha diversity, presence/absence analyses
            ⚠️ Warning: Discards data and may reduce statistical power
            """)
            
            if st.button("Apply Rarefaction", type="primary"):
                # Simple rarefaction implementation
                np.random.seed(42)
                otu_rarefied = otu_to_normalize.copy()
                
                for col in otu_rarefied.columns:
                    sample_counts = otu_rarefied[col].values
                    total = int(sample_counts.sum())
                    
                    if total >= rarefaction_depth:
                        # Create pool of reads
                        read_pool = np.repeat(np.arange(len(sample_counts)), sample_counts.astype(int))
                        # Subsample
                        subsampled = np.random.choice(read_pool, rarefaction_depth, replace=False)
                        # Count
                        rarefied_counts = np.bincount(subsampled, minlength=len(sample_counts))
                        otu_rarefied[col] = rarefied_counts
                
                st.session_state.otu_table_normalized = otu_rarefied
                st.success("✅ Rarefaction applied!")
        
        elif normalization_method == "CLR (Centered Log-Ratio)":
            st.info("""
            **CLR** applies log-ratio transformation after adding pseudocount.
            Best for: Compositional data analysis, machine learning
            """)
            
            pseudocount = st.number_input("Pseudocount", min_value=0.0, value=1.0, step=0.1)
            
            if st.button("Apply CLR Transformation", type="primary"):
                # Add pseudocount
                otu_pseudo = otu_to_normalize + pseudocount
                # Calculate geometric mean per sample
                geo_means = np.exp(np.log(otu_pseudo).mean(axis=0))
                # CLR transformation
                otu_clr = np.log(otu_pseudo.div(geo_means, axis=1))
                
                st.session_state.otu_table_normalized = otu_clr
                st.success("✅ CLR transformation applied!")
        
        else:
            st.info("Raw counts will be used (suitable for DESeq2 and other count-based methods)")
            if st.button("Use Raw Counts", type="primary"):
                st.session_state.otu_table_normalized = otu_to_normalize
                st.success("✅ Raw counts retained!")
    
    with tab4:
        st.subheader("Additional Transformations")
        
        st.markdown("""
        Apply mathematical transformations to stabilize variance or meet statistical assumptions.
        """)
        
        transformation = st.selectbox(
            "Select Transformation:",
            [
                "None",
                "Log10 (with pseudocount)",
                "Square Root",
                "Arcsine Square Root",
                "Z-score (standardization)"
            ]
        )
        
        if 'otu_table_normalized' in st.session_state:
            otu_to_transform = st.session_state.otu_table_normalized.copy()
            
            if transformation == "Log10 (with pseudocount)":
                pseudocount = st.number_input("Pseudocount", min_value=0.0, value=1.0, step=0.1, key='log_pseudo')
                
                if st.button("Apply Log10 Transformation"):
                    otu_transformed = np.log10(otu_to_transform + pseudocount)
                    st.session_state.otu_table_final = otu_transformed
                    st.success("✅ Log10 transformation applied!")
            
            elif transformation == "Square Root":
                if st.button("Apply Square Root Transformation"):
                    otu_transformed = np.sqrt(otu_to_transform)
                    st.session_state.otu_table_final = otu_transformed
                    st.success("✅ Square root transformation applied!")
            
            elif transformation == "Z-score (standardization)":
                if st.button("Apply Z-score Standardization"):
                    otu_transformed = (otu_to_transform - otu_to_transform.mean(axis=1).values.reshape(-1, 1)) / \
                                     otu_to_transform.std(axis=1).values.reshape(-1, 1)
                    st.session_state.otu_table_final = otu_transformed
                    st.success("✅ Z-score standardization applied!")
            
            else:
                st.info("No additional transformation selected")
                if st.button("Finalize Without Transformation"):
                    st.session_state.otu_table_final = otu_to_transform
                    st.success("✅ Data finalized!")
        else:
            st.warning("Please apply normalization first (in the Normalization tab)")
    
    # Summary
    st.markdown("---")
    st.subheader("📊 Preprocessing Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Original Data:**")
        st.write(f"- Samples: {len(otu_table.columns)}")
        st.write(f"- Features: {len(otu_table)}")
    
    with col2:
        if 'otu_table_filtered' in st.session_state:
            st.write("**After Filtering:**")
            st.write(f"- Samples: {len(st.session_state.otu_table_filtered.columns)}")
            st.write(f"- Features: {len(st.session_state.otu_table_filtered)}")
        else:
            st.write("**After Filtering:**")
            st.write("Not yet applied")
    
    with col3:
        if 'otu_table_final' in st.session_state:
            st.write("**Final Data:**")
            st.write(f"- Samples: {len(st.session_state.otu_table_final.columns)}")
            st.write(f"- Features: {len(st.session_state.otu_table_final)}")
            st.success("✅ Ready for analysis!")
        else:
            st.write("**Final Data:**")
            st.write("Not yet finalized")
