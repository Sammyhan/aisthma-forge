"""
Differential Abundance Module
Statistical testing for differentially abundant taxa using pydeseq2
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def run_deseq2_analysis(count_data, metadata, condition_column, reference_level, test_level):
    """
    Run DESeq2-like analysis using pydeseq2
    """
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
        
        # Prepare data
        # DESeq2 expects samples as rows, genes/features as columns
        counts_df = count_data.T
        
        # Ensure integer counts
        counts_df = counts_df.round().astype(int)
        
        # Prepare metadata
        metadata_subset = metadata.copy()
        if 'SampleID' in metadata_subset.columns:
            metadata_subset = metadata_subset.set_index('SampleID')
        
        # Keep only samples in count data
        metadata_subset = metadata_subset.loc[counts_df.index]
        
        # Ensure condition is categorical
        metadata_subset[condition_column] = metadata_subset[condition_column].astype(str)
        
        # Create DESeq2 dataset
        dds = DeseqDataSet(
            counts=counts_df,
            metadata=metadata_subset,
            design_factors=condition_column,
            refit_cooks=True,
            n_cpus=1
        )
        
        # Run DESeq2
        dds.deseq2()
        
        # Get results
        stat_res = DeseqStats(dds, contrast=[condition_column, test_level, reference_level])
        stat_res.summary()
        
        results_df = stat_res.results_df
        
        # Add additional columns
        results_df['significant'] = (results_df['padj'] < 0.05) & (np.abs(results_df['log2FoldChange']) > 1)
        
        return results_df, True
        
    except ImportError:
        st.error("pydeseq2 not available. Using alternative method.")
        return None, False
    except Exception as e:
        st.error(f"DESeq2 analysis failed: {str(e)}")
        return None, False

def run_simple_differential_test(count_data, metadata, condition_column, reference_level, test_level):
    """
    Simple differential abundance test using Mann-Whitney U and fold change
    """
    
    # Prepare metadata
    metadata_subset = metadata.copy()
    if 'SampleID' in metadata_subset.columns:
        metadata_subset = metadata_subset.set_index('SampleID')
    
    # Get sample groups
    ref_samples = metadata_subset[metadata_subset[condition_column] == reference_level].index
    test_samples = metadata_subset[metadata_subset[condition_column] == test_level].index
    
    # Filter count data
    ref_samples = [s for s in ref_samples if s in count_data.columns]
    test_samples = [s for s in test_samples if s in count_data.columns]
    
    results = []
    
    for feature in count_data.index:
        ref_counts = count_data.loc[feature, ref_samples].values
        test_counts = count_data.loc[feature, test_samples].values
        
        # Calculate means
        ref_mean = ref_counts.mean()
        test_mean = test_counts.mean()
        
        # Fold change (with pseudocount)
        fold_change = (test_mean + 1) / (ref_mean + 1)
        log2_fold_change = np.log2(fold_change)
        
        # Mann-Whitney U test
        try:
            u_stat, p_value = stats.mannwhitneyu(test_counts, ref_counts, alternative='two-sided')
        except:
            p_value = 1.0
        
        results.append({
            'Feature': feature,
            'baseMean': (ref_mean + test_mean) / 2,
            'log2FoldChange': log2_fold_change,
            'pvalue': p_value,
            f'{reference_level}_mean': ref_mean,
            f'{test_level}_mean': test_mean
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('Feature')
    
    # FDR correction (Benjamini-Hochberg)
    from scipy.stats import false_discovery_control
    try:
        results_df['padj'] = false_discovery_control(results_df['pvalue'].values)
    except:
        # Fallback: simple Bonferroni
        results_df['padj'] = results_df['pvalue'] * len(results_df)
        results_df['padj'] = results_df['padj'].clip(upper=1.0)
    
    results_df['significant'] = (results_df['padj'] < 0.05) & (np.abs(results_df['log2FoldChange']) > 1)
    
    return results_df

def render_differential_page():
    """Render the differential abundance page"""
    
    st.header("🧬 Differential Abundance Analysis")
    
    # Check for required data
    if 'otu_table' not in st.session_state:
        st.warning("⚠️ Please upload data first.")
        return
    
    # Use filtered data if available, otherwise original
    if 'otu_table_filtered' in st.session_state:
        otu_table = st.session_state.otu_table_filtered.copy()
    else:
        otu_table = st.session_state.otu_table.copy()
    
    metadata = st.session_state.metadata_filtered if 'metadata_filtered' in st.session_state else st.session_state.metadata
    
    st.markdown("""
    Identify microbial features that are significantly different in abundance between groups.
    This analysis uses **DESeq2-like** methodology designed for count data with appropriate 
    normalization and dispersion estimation.
    """)
    
    # Configuration
    st.subheader("⚙️ Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select condition column
        categorical_cols = metadata.select_dtypes(include=['object', 'category']).columns.tolist()
        if 'SampleID' in categorical_cols:
            categorical_cols.remove('SampleID')
        
        if len(categorical_cols) == 0:
            st.error("No categorical variables found in metadata.")
            return
        
        condition_column = st.selectbox(
            "Select condition variable:",
            categorical_cols,
            help="The grouping variable for comparison (e.g., Asthma_Status)"
        )
        
        # Get unique levels
        levels = metadata[condition_column].unique().tolist()
        
        if len(levels) < 2:
            st.error(f"Need at least 2 groups in {condition_column}")
            return
    
    with col2:
        reference_level = st.selectbox(
            "Reference group (control):",
            levels,
            help="The baseline group for comparison"
        )
        
        test_levels = [l for l in levels if l != reference_level]
        test_level = st.selectbox(
            "Test group:",
            test_levels,
            help="The group to compare against reference"
        )
    
    # Analysis method
    analysis_method = st.radio(
        "Select analysis method:",
        ["DESeq2 (pydeseq2)", "Simple Test (Mann-Whitney U + Fold Change)"],
        help="DESeq2 is recommended for count data; Simple test is faster but less sophisticated"
    )
    
    # Significance thresholds
    col1, col2 = st.columns(2)
    with col1:
        padj_threshold = st.number_input(
            "Adjusted p-value threshold:",
            min_value=0.001,
            max_value=0.5,
            value=0.05,
            step=0.01
        )
    
    with col2:
        lfc_threshold = st.number_input(
            "Log2 fold-change threshold:",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.5
        )
    
    # Run analysis
    if st.button("🚀 Run Differential Abundance Analysis", type="primary"):
        
        with st.spinner("Running differential abundance analysis..."):
            
            # Ensure we have integer counts for DESeq2
            count_data = otu_table.copy()
            
            # If data is normalized (values < 10), try to get raw counts
            if count_data.sum(axis=0).mean() < 10:
                st.warning("Data appears to be normalized. DESeq2 requires raw counts. Results may be suboptimal.")
                # Try to scale back to counts (rough approximation)
                count_data = (count_data * 10000).round()
            
            count_data = count_data.astype(int)
            
            if analysis_method == "DESeq2 (pydeseq2)":
                results_df, success = run_deseq2_analysis(
                    count_data,
                    metadata,
                    condition_column,
                    reference_level,
                    test_level
                )
                
                if not success or results_df is None:
                    st.info("Falling back to simple test method...")
                    results_df = run_simple_differential_test(
                        count_data,
                        metadata,
                        condition_column,
                        reference_level,
                        test_level
                    )
            else:
                results_df = run_simple_differential_test(
                    count_data,
                    metadata,
                    condition_column,
                    reference_level,
                    test_level
                )
            
            # Store results
            st.session_state.differential_results = results_df
            st.session_state.da_reference = reference_level
            st.session_state.da_test = test_level
            
            st.success("✅ Analysis complete!")
    
    # Display results
    if 'differential_results' in st.session_state:
        
        results_df = st.session_state.differential_results.copy()
        
        st.markdown("---")
        st.subheader("📊 Results")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_features = len(results_df)
            st.metric("Total Features", total_features)
        
        with col2:
            sig_features = (results_df['padj'] < padj_threshold).sum()
            st.metric("Significant (p-adj)", sig_features)
        
        with col3:
            upregulated = ((results_df['padj'] < padj_threshold) & 
                          (results_df['log2FoldChange'] > lfc_threshold)).sum()
            st.metric(f"Enriched in {st.session_state.da_test}", upregulated, delta="↑")
        
        with col4:
            downregulated = ((results_df['padj'] < padj_threshold) & 
                            (results_df['log2FoldChange'] < -lfc_threshold)).sum()
            st.metric(f"Depleted in {st.session_state.da_test}", downregulated, delta="↓")
        
        # Volcano plot
        st.markdown("---")
        st.markdown("#### Volcano Plot")
        
        # Prepare data for plotting
        plot_df = results_df.copy()
        plot_df['-log10(padj)'] = -np.log10(plot_df['padj'].replace(0, 1e-300))
        plot_df['Feature_name'] = plot_df.index
        
        # Categorize features
        plot_df['Category'] = 'Not Significant'
        plot_df.loc[(plot_df['padj'] < padj_threshold) & 
                   (plot_df['log2FoldChange'] > lfc_threshold), 'Category'] = f'Enriched in {st.session_state.da_test}'
        plot_df.loc[(plot_df['padj'] < padj_threshold) & 
                   (plot_df['log2FoldChange'] < -lfc_threshold), 'Category'] = f'Depleted in {st.session_state.da_test}'
        
        fig = px.scatter(
            plot_df,
            x='log2FoldChange',
            y='-log10(padj)',
            color='Category',
            hover_data=['Feature_name'],
            title=f"Volcano Plot: {st.session_state.da_test} vs {st.session_state.da_reference}",
            color_discrete_map={
                'Not Significant': 'lightgray',
                f'Enriched in {st.session_state.da_test}': 'red',
                f'Depleted in {st.session_state.da_test}': 'blue'
            }
        )
        
        # Add threshold lines
        fig.add_hline(y=-np.log10(padj_threshold), line_dash="dash", line_color="gray")
        fig.add_vline(x=lfc_threshold, line_dash="dash", line_color="gray")
        fig.add_vline(x=-lfc_threshold, line_dash="dash", line_color="gray")
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top features table
        st.markdown("---")
        st.markdown("#### Top Differentially Abundant Features")
        
        # Filter significant features
        sig_results = results_df[
            (results_df['padj'] < padj_threshold) & 
            (np.abs(results_df['log2FoldChange']) > lfc_threshold)
        ].copy()
        
        if len(sig_results) > 0:
            # Sort by adjusted p-value
            sig_results = sig_results.sort_values('padj')
            
            # Display top 20
            display_cols = ['log2FoldChange', 'baseMean', 'pvalue', 'padj']
            if f'{st.session_state.da_reference}_mean' in sig_results.columns:
                display_cols.insert(1, f'{st.session_state.da_reference}_mean')
                display_cols.insert(2, f'{st.session_state.da_test}_mean')
            
            st.dataframe(
                sig_results[display_cols].head(20).style.format({
                    'log2FoldChange': '{:.3f}',
                    'baseMean': '{:.2f}',
                    'pvalue': '{:.2e}',
                    'padj': '{:.2e}'
                }),
                use_container_width=True
            )
            
            # Download full results
            st.markdown("---")
            csv = sig_results.to_csv()
            st.download_button(
                label="📥 Download Significant Features (CSV)",
                data=csv,
                file_name=f"differential_abundance_{st.session_state.da_test}_vs_{st.session_state.da_reference}.csv",
                mime="text/csv"
            )
            
            # Feature details
            st.markdown("---")
            st.markdown("#### Explore Individual Features")
            
            feature_to_plot = st.selectbox(
                "Select feature to visualize:",
                sig_results.head(50).index.tolist()
            )
            
            if feature_to_plot:
                # Get counts for this feature
                feature_counts = otu_table.loc[feature_to_plot, :]
                
                # Merge with metadata
                plot_data = pd.DataFrame({
                    'SampleID': feature_counts.index,
                    'Abundance': feature_counts.values
                })
                
                if 'SampleID' in metadata.columns:
                    plot_data = plot_data.merge(metadata[['SampleID', condition_column]], on='SampleID')
                else:
                    metadata_temp = metadata.copy()
                    metadata_temp['SampleID'] = metadata_temp.index
                    plot_data = plot_data.merge(metadata_temp[['SampleID', condition_column]], on='SampleID')
                
                # Box plot
                fig = px.box(
                    plot_data,
                    x=condition_column,
                    y='Abundance',
                    color=condition_column,
                    points='all',
                    title=f"Abundance of {feature_to_plot}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show statistics for this feature
                feature_stats = results_df.loc[feature_to_plot]
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Log2 Fold Change", f"{feature_stats['log2FoldChange']:.3f}")
                
                with col2:
                    st.metric("Adjusted p-value", f"{feature_stats['padj']:.2e}")
                
                with col3:
                    st.metric("Base Mean", f"{feature_stats['baseMean']:.2f}")
        
        else:
            st.info(f"No features meet the significance criteria (padj < {padj_threshold}, |log2FC| > {lfc_threshold})")
            
            # Still show all results
            st.markdown("#### All Features (sorted by p-value)")
            display_cols = ['log2FoldChange', 'baseMean', 'pvalue', 'padj']
            st.dataframe(
                results_df.sort_values('pvalue')[display_cols].head(20).style.format({
                    'log2FoldChange': '{:.3f}',
                    'baseMean': '{:.2f}',
                    'pvalue': '{:.2e}',
                    'padj': '{:.2e}'
                })
            )
    
    else:
        st.info("👆 Configure parameters above and click 'Run Analysis' to begin.")
