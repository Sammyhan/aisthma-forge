"""
Reports and Export Module
Generate comprehensive reports and export results
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json

def generate_comprehensive_report():
    """Generate a comprehensive analysis report"""
    
    report = {
        'metadata': {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'platform': 'AIsthma Forge v1.0',
            'analysis_type': 'Microbiome Analysis for Asthma Research'
        },
        'data_summary': {},
        'preprocessing': {},
        'diversity': {},
        'differential_abundance': {},
        'machine_learning': {},
        'gene_hypotheses': {}
    }
    
    # Data summary
    if 'otu_table' in st.session_state:
        report['data_summary'] = {
            'total_samples': len(st.session_state.otu_table.columns),
            'total_features': len(st.session_state.otu_table),
            'total_reads': int(st.session_state.otu_table.sum().sum())
        }
    
    # Preprocessing
    if 'otu_table_final' in st.session_state:
        report['preprocessing'] = {
            'final_samples': len(st.session_state.otu_table_final.columns),
            'final_features': len(st.session_state.otu_table_final),
            'filtering_applied': 'otu_table_filtered' in st.session_state
        }
    
    # Diversity
    if 'alpha_diversity' in st.session_state:
        alpha_div = st.session_state.alpha_diversity
        report['diversity']['alpha'] = {
            'mean_richness': float(alpha_div['Richness'].mean()),
            'mean_shannon': float(alpha_div['Shannon'].mean()),
            'mean_simpson': float(alpha_div['Simpson'].mean())
        }
    
    # Differential abundance
    if 'differential_results' in st.session_state:
        diff_results = st.session_state.differential_results
        sig_features = diff_results[diff_results['significant'] == True]
        
        report['differential_abundance'] = {
            'total_tested': len(diff_results),
            'significant_features': len(sig_features),
            'enriched': int(((sig_features['log2FoldChange'] > 1)).sum()),
            'depleted': int(((sig_features['log2FoldChange'] < -1)).sum()),
            'comparison': f"{st.session_state.get('da_test', 'Test')} vs {st.session_state.get('da_reference', 'Reference')}"
        }
    
    # Machine learning
    if 'ml_metrics' in st.session_state:
        metrics = st.session_state.ml_metrics
        report['machine_learning'] = {
            'model_type': st.session_state.get('ml_model_type', 'Unknown'),
            'target_variable': st.session_state.get('ml_target', 'Unknown'),
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1']),
            'roc_auc': float(metrics['roc_auc'])
        }
    
    # Gene hypotheses
    if 'gene_hypotheses' in st.session_state:
        hypotheses = st.session_state.gene_hypotheses
        report['gene_hypotheses'] = {
            'total_hypotheses': len(hypotheses),
            'high_priority': int((hypotheses['priority'] == 1).sum()),
            'pathways_affected': int(hypotheses['pathway'].nunique()),
            'unique_genera': int(hypotheses['genus'].nunique())
        }
    
    return report

def render_reports_page():
    """Render the reports and export page"""
    
    st.header("📄 Reports & Export")
    
    st.markdown("""
    Generate comprehensive reports and export your analysis results in various formats.
    """)
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Analysis Summary",
        "📥 Export Data",
        "📋 Generate Report"
    ])
    
    with tab1:
        st.subheader("Analysis Summary")
        
        # Generate summary
        report = generate_comprehensive_report()
        
        # Display metadata
        st.markdown("#### Analysis Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Platform:** {report['metadata']['platform']}")
            st.write(f"**Generated:** {report['metadata']['generated_at']}")
        
        with col2:
            st.write(f"**Analysis Type:** {report['metadata']['analysis_type']}")
        
        # Data summary
        if report['data_summary']:
            st.markdown("---")
            st.markdown("#### Data Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Samples", report['data_summary'].get('total_samples', 'N/A'))
            
            with col2:
                st.metric("Total Features", report['data_summary'].get('total_features', 'N/A'))
            
            with col3:
                total_reads = report['data_summary'].get('total_reads', 0)
                st.metric("Total Reads", f"{total_reads:,}" if total_reads else 'N/A')
        
        # Preprocessing
        if report['preprocessing']:
            st.markdown("---")
            st.markdown("#### Preprocessing")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Final Samples", report['preprocessing'].get('final_samples', 'N/A'))
                st.metric("Final Features", report['preprocessing'].get('final_features', 'N/A'))
            
            with col2:
                filtering = "✅ Yes" if report['preprocessing'].get('filtering_applied') else "❌ No"
                st.write(f"**Filtering Applied:** {filtering}")
        
        # Diversity
        if 'alpha' in report['diversity']:
            st.markdown("---")
            st.markdown("#### Diversity Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Richness", f"{report['diversity']['alpha']['mean_richness']:.1f}")
            
            with col2:
                st.metric("Mean Shannon", f"{report['diversity']['alpha']['mean_shannon']:.3f}")
            
            with col3:
                st.metric("Mean Simpson", f"{report['diversity']['alpha']['mean_simpson']:.3f}")
        
        # Differential abundance
        if report['differential_abundance']:
            st.markdown("---")
            st.markdown("#### Differential Abundance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Features Tested", report['differential_abundance']['total_tested'])
            
            with col2:
                st.metric("Significant", report['differential_abundance']['significant_features'])
            
            with col3:
                st.metric("Enriched", report['differential_abundance']['enriched'], delta="↑")
            
            with col4:
                st.metric("Depleted", report['differential_abundance']['depleted'], delta="↓")
            
            st.write(f"**Comparison:** {report['differential_abundance']['comparison']}")
        
        # Machine learning
        if report['machine_learning']:
            st.markdown("---")
            st.markdown("#### Machine Learning")
            
            st.write(f"**Model:** {report['machine_learning']['model_type']}")
            st.write(f"**Target:** {report['machine_learning']['target_variable']}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Accuracy", f"{report['machine_learning']['accuracy']:.3f}")
            
            with col2:
                st.metric("Precision", f"{report['machine_learning']['precision']:.3f}")
            
            with col3:
                st.metric("Recall", f"{report['machine_learning']['recall']:.3f}")
            
            with col4:
                st.metric("F1 Score", f"{report['machine_learning']['f1_score']:.3f}")
            
            with col5:
                st.metric("ROC AUC", f"{report['machine_learning']['roc_auc']:.3f}")
        
        # Gene hypotheses
        if report['gene_hypotheses']:
            st.markdown("---")
            st.markdown("#### Gene Hypotheses")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Hypotheses", report['gene_hypotheses']['total_hypotheses'])
            
            with col2:
                st.metric("High Priority", report['gene_hypotheses']['high_priority'])
            
            with col3:
                st.metric("Pathways", report['gene_hypotheses']['pathways_affected'])
            
            with col4:
                st.metric("Unique Genera", report['gene_hypotheses']['unique_genera'])
        
        # Export summary as JSON
        st.markdown("---")
        
        json_report = json.dumps(report, indent=2)
        st.download_button(
            label="📥 Download Summary (JSON)",
            data=json_report,
            file_name=f"aisthma_forge_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with tab2:
        st.subheader("Export Analysis Results")
        
        st.markdown("Download individual analysis results in CSV format:")
        
        # OTU table
        if 'otu_table_final' in st.session_state:
            st.markdown("#### Processed OTU Table")
            csv = st.session_state.otu_table_final.to_csv()
            st.download_button(
                label="📥 Download Processed OTU Table (CSV)",
                data=csv,
                file_name="otu_table_processed.csv",
                mime="text/csv"
            )
        
        # Alpha diversity
        if 'alpha_diversity' in st.session_state:
            st.markdown("#### Alpha Diversity Metrics")
            csv = st.session_state.alpha_diversity.to_csv()
            st.download_button(
                label="📥 Download Alpha Diversity (CSV)",
                data=csv,
                file_name="alpha_diversity.csv",
                mime="text/csv"
            )
        
        # Beta diversity
        if 'beta_diversity' in st.session_state:
            st.markdown("#### Beta Diversity Distance Matrix")
            csv = st.session_state.beta_diversity.to_csv()
            st.download_button(
                label="📥 Download Beta Diversity (CSV)",
                data=csv,
                file_name="beta_diversity_distances.csv",
                mime="text/csv"
            )
        
        # Differential abundance
        if 'differential_results' in st.session_state:
            st.markdown("#### Differential Abundance Results")
            csv = st.session_state.differential_results.to_csv()
            st.download_button(
                label="📥 Download Differential Abundance (CSV)",
                data=csv,
                file_name="differential_abundance_results.csv",
                mime="text/csv"
            )
        
        # ML predictions
        if 'ml_y_pred' in st.session_state:
            st.markdown("#### Machine Learning Predictions")
            
            predictions_df = pd.DataFrame({
                'Sample': st.session_state.ml_X_test.index,
                'Actual': st.session_state.ml_y_test.values,
                'Predicted': st.session_state.ml_y_pred
            })
            
            # Add probabilities
            if 'ml_y_pred_proba' in st.session_state:
                proba = st.session_state.ml_y_pred_proba
                for i, class_name in enumerate(st.session_state.ml_model.classes_):
                    predictions_df[f'Prob_{class_name}'] = proba[:, i]
            
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="📥 Download ML Predictions (CSV)",
                data=csv,
                file_name="ml_predictions.csv",
                mime="text/csv"
            )
        
        # Gene hypotheses
        if 'gene_hypotheses' in st.session_state:
            st.markdown("#### Gene Hypotheses")
            csv = st.session_state.gene_hypotheses.to_csv(index=False)
            st.download_button(
                label="📥 Download Gene Hypotheses (CSV)",
                data=csv,
                file_name="gene_hypotheses.csv",
                mime="text/csv"
            )
        
        # Export all data as ZIP
        st.markdown("---")
        st.markdown("#### Export All Results")
        
        if st.button("📦 Prepare Complete Export Package"):
            st.info("This feature would create a ZIP file with all results. Implementation requires additional file handling.")
    
    with tab3:
        st.subheader("Generate Comprehensive Report")
        
        st.markdown("""
        Create a detailed markdown report summarizing all analysis steps and findings.
        """)
        
        # Report options
        include_plots = st.checkbox("Include plot descriptions", value=True)
        include_methods = st.checkbox("Include methods section", value=True)
        include_recommendations = st.checkbox("Include recommendations", value=True)
        
        if st.button("📄 Generate Report", type="primary"):
            
            with st.spinner("Generating comprehensive report..."):
                
                report_md = f"""# AIsthma Forge Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Platform:** AIsthma Forge v1.0  
**Analysis Type:** Microbiome Analysis for Asthma Research

---

## Executive Summary

This report presents a comprehensive analysis of microbiome sequencing data to identify 
microbial features and functional pathways associated with asthma pathogenesis.

"""
                
                # Data summary
                if 'otu_table' in st.session_state:
                    n_samples = len(st.session_state.otu_table.columns)
                    n_features = len(st.session_state.otu_table)
                    
                    report_md += f"""
## Data Overview

- **Total Samples:** {n_samples}
- **Total Features:** {n_features}
- **Total Reads:** {int(st.session_state.otu_table.sum().sum()):,}

"""
                
                # Preprocessing
                if 'otu_table_final' in st.session_state:
                    final_samples = len(st.session_state.otu_table_final.columns)
                    final_features = len(st.session_state.otu_table_final)
                    
                    report_md += f"""
## Preprocessing

Data underwent quality control and preprocessing:

- **Samples after filtering:** {final_samples}
- **Features after filtering:** {final_features}
- **Normalization:** Applied
- **Transformation:** Applied

"""
                
                # Diversity analysis
                if 'alpha_diversity' in st.session_state:
                    alpha_div = st.session_state.alpha_diversity
                    
                    report_md += f"""
## Diversity Analysis

### Alpha Diversity

Within-sample diversity metrics:

- **Mean Richness:** {alpha_div['Richness'].mean():.1f} ± {alpha_div['Richness'].std():.1f}
- **Mean Shannon Index:** {alpha_div['Shannon'].mean():.3f} ± {alpha_div['Shannon'].std():.3f}
- **Mean Simpson Index:** {alpha_div['Simpson'].mean():.3f} ± {alpha_div['Simpson'].std():.3f}

"""
                
                # Differential abundance
                if 'differential_results' in st.session_state:
                    diff_results = st.session_state.differential_results
                    sig_features = diff_results[diff_results['significant'] == True]
                    
                    report_md += f"""
## Differential Abundance Analysis

Comparison: **{st.session_state.get('da_test', 'Test')} vs {st.session_state.get('da_reference', 'Reference')}**

- **Features tested:** {len(diff_results)}
- **Significant features:** {len(sig_features)} (padj < 0.05, |log2FC| > 1)
- **Enriched in {st.session_state.get('da_test', 'Test')}:** {((sig_features['log2FoldChange'] > 1)).sum()}
- **Depleted in {st.session_state.get('da_test', 'Test')}:** {((sig_features['log2FoldChange'] < -1)).sum()}

### Top 10 Differentially Abundant Features

"""
                    
                    for idx, (feature, row) in enumerate(sig_features.head(10).iterrows(), 1):
                        report_md += f"{idx}. **{feature}**\n"
                        report_md += f"   - Log2 Fold Change: {row['log2FoldChange']:.3f}\n"
                        report_md += f"   - Adjusted p-value: {row['padj']:.2e}\n\n"
                
                # Machine learning
                if 'ml_metrics' in st.session_state:
                    metrics = st.session_state.ml_metrics
                    
                    report_md += f"""
## Machine Learning Classification

**Model:** {st.session_state.get('ml_model_type', 'Unknown')}  
**Target Variable:** {st.session_state.get('ml_target', 'Unknown')}

### Performance Metrics

- **Accuracy:** {metrics['accuracy']:.3f}
- **Precision:** {metrics['precision']:.3f}
- **Recall:** {metrics['recall']:.3f}
- **F1 Score:** {metrics['f1']:.3f}
- **ROC AUC:** {metrics['roc_auc']:.3f}

The model demonstrates {'strong' if metrics['accuracy'] > 0.8 else 'moderate' if metrics['accuracy'] > 0.7 else 'limited'} 
predictive performance in classifying samples based on microbial profiles.

"""
                
                # Gene hypotheses
                if 'gene_hypotheses' in st.session_state:
                    hypotheses = st.session_state.gene_hypotheses
                    
                    report_md += f"""
## Gene Hypotheses & Functional Predictions

Generated **{len(hypotheses)} hypotheses** linking microbial alterations to asthma pathogenesis.

### High Priority Hypotheses

"""
                    
                    high_priority = hypotheses[hypotheses['priority'] == 1].head(5)
                    
                    for idx, row in high_priority.iterrows():
                        report_md += f"""
#### {row['genus']} - {row['pathway_description']}

**Hypothesis:** {row['hypothesis']}

**Direction:** {row['direction']}  
**Pathway:** {row['pathway']}

---

"""
                    
                    # Pathway summary
                    pathway_counts = hypotheses['pathway'].value_counts()
                    
                    report_md += """
### Pathways Affected

"""
                    
                    for pathway, count in pathway_counts.items():
                        report_md += f"- **{pathway.replace('_', ' ').title()}:** {count} features\n"
                
                # Methods
                if include_methods:
                    report_md += """

---

## Methods

### Data Processing

Raw sequencing data (OTU/ASV counts) were processed using the AIsthma Forge pipeline:

1. **Quality Control:** Sample depth and feature prevalence filtering
2. **Normalization:** Total sum scaling (TSS) or appropriate method
3. **Transformation:** Log transformation with pseudocount

### Statistical Analysis

- **Alpha Diversity:** Richness, Shannon, Simpson, and Evenness indices
- **Beta Diversity:** Bray-Curtis dissimilarity with PERMANOVA testing
- **Differential Abundance:** DESeq2-like methodology for count data
- **Multiple Testing Correction:** Benjamini-Hochberg FDR

### Machine Learning

- **Algorithm:** Random Forest / XGBoost ensemble methods
- **Validation:** Train-test split with stratified cross-validation
- **Explainability:** SHAP (SHapley Additive exPlanations) values

### Functional Prediction

Gene hypotheses generated by integrating:
- Differential abundance results
- Machine learning feature importance
- Known asthma-related microbial pathways
- Taxonomic annotations

"""
                
                # Recommendations
                if include_recommendations:
                    report_md += """

---

## Recommendations

### Research Priorities

1. **Validate Top Hypotheses:** Conduct targeted functional metagenomics on high-priority features
2. **Pathway Analysis:** Investigate specific gene expression in identified pathways
3. **Longitudinal Studies:** Track microbiome changes over time in asthma cohorts
4. **Mechanistic Studies:** Explore gut-lung axis mechanisms in animal models

### Clinical Implications

1. **Biomarker Development:** Evaluate top features as diagnostic biomarkers
2. **Therapeutic Targets:** Consider microbiome-based interventions
3. **Personalized Medicine:** Use ML models for risk stratification
4. **Probiotic Development:** Target beneficial taxa deficits

### Technical Considerations

1. **Sample Size:** Increase cohort size for validation
2. **Confounders:** Control for age, diet, antibiotics, geography
3. **Multi-omics:** Integrate metabolomics and transcriptomics
4. **Replication:** Validate in independent cohorts

"""
                
                report_md += """

---

## Conclusion

This analysis identified significant microbial alterations associated with asthma, 
providing actionable hypotheses for functional gene changes and potential therapeutic targets. 
The integration of differential abundance, machine learning, and functional prediction 
offers a comprehensive view of microbiome dysbiosis in asthma pathogenesis.

---

*Report generated by AIsthma Forge - Open-source microbiome analysis platform for asthma research*

"""
                
                st.session_state.final_report = report_md
                
                st.success("✅ Report generated successfully!")
        
        # Display and download report
        if 'final_report' in st.session_state:
            
            st.markdown("---")
            st.markdown("### Report Preview")
            
            with st.expander("View Full Report", expanded=False):
                st.markdown(st.session_state.final_report)
            
            # Download options
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📥 Download Report (Markdown)",
                    data=st.session_state.final_report,
                    file_name=f"aisthma_forge_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            
            with col2:
                st.info("PDF export available via markdown-to-PDF conversion tools")
