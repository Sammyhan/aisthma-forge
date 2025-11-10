"""
Predictive Modeling Module
Generate hypotheses for gene alterations linked to asthma pathogenesis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Known asthma-related microbial genes and pathways
ASTHMA_GENE_DATABASE = {
    'virulence_factors': {
        'description': 'Bacterial virulence factors that may trigger inflammation',
        'genes': [
            'lipopolysaccharide biosynthesis',
            'peptidoglycan synthesis',
            'flagellar assembly',
            'type III secretion system',
            'adhesion proteins',
            'toxin production'
        ],
        'taxa': ['Moraxella', 'Haemophilus', 'Streptococcus', 'Staphylococcus']
    },
    'butyrate_production': {
        'description': 'Short-chain fatty acid production (anti-inflammatory)',
        'genes': [
            'butyryl-CoA dehydrogenase',
            'butyrate kinase',
            'acetate-CoA transferase',
            'butyrate-acetoacetate CoA-transferase'
        ],
        'taxa': ['Faecalibacterium', 'Roseburia', 'Eubacterium', 'Clostridium']
    },
    'lps_biosynthesis': {
        'description': 'Lipopolysaccharide biosynthesis (pro-inflammatory)',
        'genes': [
            'lpxA', 'lpxB', 'lpxC', 'lpxD',
            'kdsA', 'kdsB',
            'waaA', 'waaC', 'waaF'
        ],
        'taxa': ['Proteobacteria', 'Bacteroidetes']
    },
    'histamine_production': {
        'description': 'Histamine metabolism (allergic response)',
        'genes': [
            'histidine decarboxylase',
            'histamine N-methyltransferase',
            'diamine oxidase'
        ],
        'taxa': ['Lactobacillus', 'Enterococcus', 'Escherichia']
    },
    'immune_modulation': {
        'description': 'Immune system modulation pathways',
        'genes': [
            'polysaccharide A biosynthesis',
            'exopolysaccharide production',
            'capsular polysaccharide',
            'peptidoglycan recognition proteins'
        ],
        'taxa': ['Bacteroides', 'Bifidobacterium', 'Lactobacillus']
    }
}

def extract_taxa_from_features(features):
    """Extract taxonomic information from feature names"""
    
    taxa_info = []
    
    for feature in features:
        # Try to parse taxonomy string
        if '|' in feature:
            parts = feature.split('|')
            taxa_dict = {
                'feature': feature,
                'kingdom': parts[0] if len(parts) > 0 else 'Unknown',
                'phylum': parts[1] if len(parts) > 1 else 'Unknown',
                'class': parts[2] if len(parts) > 2 else 'Unknown',
                'order': parts[3] if len(parts) > 3 else 'Unknown',
                'family': parts[4] if len(parts) > 4 else 'Unknown',
                'genus': parts[5] if len(parts) > 5 else 'Unknown'
            }
        else:
            taxa_dict = {
                'feature': feature,
                'kingdom': 'Unknown',
                'phylum': 'Unknown',
                'class': 'Unknown',
                'order': 'Unknown',
                'family': 'Unknown',
                'genus': feature
            }
        
        taxa_info.append(taxa_dict)
    
    return pd.DataFrame(taxa_info)

def generate_gene_hypotheses(differential_results, shap_importance=None):
    """Generate hypotheses for gene alterations based on differential abundance and ML results"""
    
    hypotheses = []
    
    # Extract significant features
    sig_features = differential_results[differential_results['significant'] == True].copy()
    sig_features = sig_features.sort_values('padj')
    
    # Extract taxa information
    taxa_df = extract_taxa_from_features(sig_features.index.tolist())
    
    # Merge with differential results
    sig_features_with_taxa = sig_features.reset_index()
    sig_features_with_taxa = sig_features_with_taxa.merge(
        taxa_df, 
        left_on='Feature' if 'Feature' in sig_features_with_taxa.columns else sig_features_with_taxa.columns[0],
        right_on='feature',
        how='left'
    )
    
    # Generate hypotheses based on known pathways
    for idx, row in sig_features_with_taxa.iterrows():
        
        genus = row['genus'] if 'genus' in row else 'Unknown'
        log2fc = row['log2FoldChange']
        padj = row['padj']
        
        # Check against known asthma-related taxa and pathways
        for pathway_name, pathway_info in ASTHMA_GENE_DATABASE.items():
            
            # Check if genus matches known taxa for this pathway
            for known_taxon in pathway_info['taxa']:
                if known_taxon.lower() in genus.lower():
                    
                    # Determine direction
                    if log2fc > 0:
                        direction = "enriched"
                        impact = "increased"
                    else:
                        direction = "depleted"
                        impact = "decreased"
                    
                    # Generate hypothesis
                    hypothesis = {
                        'feature': row['feature'] if 'feature' in row else row.get('Feature', 'Unknown'),
                        'genus': genus,
                        'pathway': pathway_name,
                        'pathway_description': pathway_info['description'],
                        'direction': direction,
                        'log2FoldChange': log2fc,
                        'padj': padj,
                        'hypothesis': f"{genus} is {direction} in asthma samples, suggesting {impact} "
                                    f"{pathway_info['description'].lower()}. "
                                    f"This may involve genes: {', '.join(pathway_info['genes'][:3])}.",
                        'confidence': 'High' if padj < 0.01 else 'Medium' if padj < 0.05 else 'Low',
                        'priority': 1 if padj < 0.01 and abs(log2fc) > 2 else 2 if padj < 0.05 else 3
                    }
                    
                    hypotheses.append(hypothesis)
                    break
    
    # If we have SHAP importance, incorporate it
    if shap_importance is not None:
        shap_df = extract_taxa_from_features(shap_importance.index.tolist())
        
        for idx, row in shap_df.iterrows():
            genus = row['genus']
            
            # Check if this is a high-importance feature
            if row['feature'] in shap_importance.head(20).index:
                
                for pathway_name, pathway_info in ASTHMA_GENE_DATABASE.items():
                    for known_taxon in pathway_info['taxa']:
                        if known_taxon.lower() in genus.lower():
                            
                            # Check if not already in hypotheses
                            if not any(h['feature'] == row['feature'] for h in hypotheses):
                                
                                hypothesis = {
                                    'feature': row['feature'],
                                    'genus': genus,
                                    'pathway': pathway_name,
                                    'pathway_description': pathway_info['description'],
                                    'direction': 'ML-identified',
                                    'log2FoldChange': np.nan,
                                    'padj': np.nan,
                                    'hypothesis': f"{genus} identified as important predictor by machine learning, "
                                                f"potentially linked to {pathway_info['description'].lower()}. "
                                                f"Candidate genes: {', '.join(pathway_info['genes'][:3])}.",
                                    'confidence': 'ML-derived',
                                    'priority': 2
                                }
                                
                                hypotheses.append(hypothesis)
                                break
    
    return pd.DataFrame(hypotheses)

def render_predictive_page():
    """Render the predictive modeling page"""
    
    st.header("🔬 Predictive Modeling & Gene Hypotheses")
    
    st.markdown("""
    This module integrates differential abundance and machine learning results to generate 
    **ranked hypotheses** for microbial gene alterations linked to asthma pathogenesis.
    
    The system identifies:
    - **Virulence factors** in pathogenic taxa (e.g., Moraxella, Haemophilus)
    - **Butyrate pathway deficits** in beneficial taxa (e.g., Faecalibacterium)
    - **LPS biosynthesis** genes driving inflammation
    - **Histamine metabolism** affecting allergic responses
    - **Immune modulation** pathways in the gut-lung axis
    """)
    
    # Check for required data
    if 'differential_results' not in st.session_state:
        st.warning("⚠️ Please run differential abundance analysis first")
        return
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs([
        "🧬 Gene Hypotheses",
        "🗺️ Pathway Mapping",
        "📊 Integrated Analysis"
    ])
    
    with tab1:
        st.subheader("Generated Gene Hypotheses")
        
        # Get SHAP importance if available
        shap_importance = None
        if 'shap_values' in st.session_state:
            X_test = st.session_state.ml_X_test
            shap_values = st.session_state.shap_values
            
            if isinstance(shap_values, list):
                shap_values_use = shap_values[1]
            else:
                shap_values_use = shap_values
            
            mean_shap = np.abs(shap_values_use).mean(axis=0)
            shap_importance = pd.Series(mean_shap, index=X_test.columns).sort_values(ascending=False)
        
        # Generate hypotheses
        with st.spinner("Generating hypotheses..."):
            hypotheses_df = generate_gene_hypotheses(
                st.session_state.differential_results,
                shap_importance
            )
        
        if len(hypotheses_df) > 0:
            
            st.session_state.gene_hypotheses = hypotheses_df
            
            st.success(f"✅ Generated {len(hypotheses_df)} gene hypotheses")
            
            # Filter options
            col1, col2 = st.columns(2)
            
            with col1:
                pathway_filter = st.multiselect(
                    "Filter by pathway:",
                    hypotheses_df['pathway'].unique().tolist(),
                    default=hypotheses_df['pathway'].unique().tolist()
                )
            
            with col2:
                priority_filter = st.multiselect(
                    "Filter by priority:",
                    sorted(hypotheses_df['priority'].unique().tolist()),
                    default=sorted(hypotheses_df['priority'].unique().tolist())
                )
            
            # Apply filters
            filtered_hypotheses = hypotheses_df[
                (hypotheses_df['pathway'].isin(pathway_filter)) &
                (hypotheses_df['priority'].isin(priority_filter))
            ].sort_values('priority')
            
            # Display hypotheses
            st.markdown("---")
            st.markdown("#### Top Hypotheses")
            
            for idx, row in filtered_hypotheses.head(10).iterrows():
                
                with st.expander(f"**{row['genus']}** - {row['pathway_description']}", expanded=(idx < 3)):
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write("**Hypothesis:**")
                        st.info(row['hypothesis'])
                    
                    with col2:
                        st.metric("Priority", row['priority'])
                        st.metric("Direction", row['direction'])
                    
                    with col3:
                        if not pd.isna(row['log2FoldChange']):
                            st.metric("Log2FC", f"{row['log2FoldChange']:.2f}")
                            st.metric("Adj. p-value", f"{row['padj']:.2e}")
                        else:
                            st.metric("Source", "ML Model")
                    
                    st.write(f"**Feature:** `{row['feature']}`")
                    st.write(f"**Pathway:** {row['pathway']}")
            
            # Summary by pathway
            st.markdown("---")
            st.markdown("#### Hypotheses by Pathway")
            
            pathway_summary = hypotheses_df.groupby('pathway').agg({
                'feature': 'count',
                'priority': 'mean'
            }).reset_index()
            pathway_summary.columns = ['Pathway', 'Count', 'Avg Priority']
            pathway_summary = pathway_summary.sort_values('Count', ascending=False)
            
            fig = px.bar(
                pathway_summary,
                x='Pathway',
                y='Count',
                title='Number of Hypotheses by Pathway',
                color='Avg Priority',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Download hypotheses
            st.markdown("---")
            csv = filtered_hypotheses.to_csv(index=False)
            st.download_button(
                label="📥 Download Gene Hypotheses (CSV)",
                data=csv,
                file_name="gene_hypotheses_asthma.csv",
                mime="text/csv"
            )
        
        else:
            st.warning("""
            No hypotheses generated. This may be because:
            - Significant features don't match known asthma-related taxa
            - Feature names don't contain taxonomic information
            - Try adjusting significance thresholds in differential abundance analysis
            """)
            
            st.info("""
            **Known asthma-related taxa in database:**
            - Moraxella, Haemophilus, Streptococcus (virulence)
            - Faecalibacterium, Roseburia, Eubacterium (butyrate production)
            - Bacteroides, Bifidobacterium (immune modulation)
            """)
    
    with tab2:
        st.subheader("Pathway Mapping")
        
        st.markdown("""
        Explore the biological pathways and gene families associated with asthma-related microbiome changes.
        """)
        
        # Display pathway database
        for pathway_name, pathway_info in ASTHMA_GENE_DATABASE.items():
            
            with st.expander(f"**{pathway_name.replace('_', ' ').title()}**"):
                
                st.write(f"**Description:** {pathway_info['description']}")
                
                st.write("**Associated Genes:**")
                for gene in pathway_info['genes']:
                    st.write(f"- {gene}")
                
                st.write("**Key Taxa:**")
                for taxon in pathway_info['taxa']:
                    st.write(f"- *{taxon}*")
                
                # Check if any of these taxa are in our results
                if 'differential_results' in st.session_state:
                    diff_results = st.session_state.differential_results
                    
                    matching_features = []
                    for feature in diff_results.index:
                        for taxon in pathway_info['taxa']:
                            if taxon.lower() in feature.lower():
                                matching_features.append(feature)
                                break
                    
                    if len(matching_features) > 0:
                        st.success(f"✅ {len(matching_features)} features in your data match this pathway")
                        
                        with st.expander("Show matching features"):
                            for feature in matching_features[:10]:
                                fc = diff_results.loc[feature, 'log2FoldChange']
                                padj = diff_results.loc[feature, 'padj']
                                st.write(f"- {feature} (log2FC: {fc:.2f}, padj: {padj:.2e})")
                    else:
                        st.info("No matching features found in your data")
    
    with tab3:
        st.subheader("Integrated Analysis Dashboard")
        
        st.markdown("""
        Comprehensive view combining differential abundance, machine learning, and functional predictions.
        """)
        
        if 'gene_hypotheses' not in st.session_state or len(st.session_state.gene_hypotheses) == 0:
            st.info("Generate hypotheses in the 'Gene Hypotheses' tab first")
            return
        
        hypotheses_df = st.session_state.gene_hypotheses
        
        # Summary metrics
        st.markdown("#### Summary Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Hypotheses", len(hypotheses_df))
        
        with col2:
            high_priority = (hypotheses_df['priority'] == 1).sum()
            st.metric("High Priority", high_priority)
        
        with col3:
            unique_genera = hypotheses_df['genus'].nunique()
            st.metric("Unique Genera", unique_genera)
        
        with col4:
            pathways = hypotheses_df['pathway'].nunique()
            st.metric("Pathways Affected", pathways)
        
        # Network-style visualization
        st.markdown("---")
        st.markdown("#### Taxa-Pathway Network")
        
        # Create network data
        network_data = hypotheses_df.groupby(['genus', 'pathway']).size().reset_index(name='count')
        
        # Sunburst chart
        fig = px.sunburst(
            network_data,
            path=['pathway', 'genus'],
            values='count',
            title='Taxa-Pathway Relationships'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Enrichment vs Depletion
        st.markdown("---")
        st.markdown("#### Enrichment vs Depletion by Pathway")
        
        direction_summary = hypotheses_df[hypotheses_df['direction'].isin(['enriched', 'depleted'])].copy()
        
        if len(direction_summary) > 0:
            direction_counts = direction_summary.groupby(['pathway', 'direction']).size().reset_index(name='count')
            
            fig = px.bar(
                direction_counts,
                x='pathway',
                y='count',
                color='direction',
                barmode='group',
                title='Feature Direction by Pathway',
                color_discrete_map={'enriched': 'red', 'depleted': 'blue'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top genera
        st.markdown("---")
        st.markdown("#### Most Frequently Implicated Genera")
        
        genus_counts = hypotheses_df['genus'].value_counts().head(15)
        
        fig = px.bar(
            x=genus_counts.values,
            y=genus_counts.index,
            orientation='h',
            title='Top 15 Genera in Hypotheses',
            labels={'x': 'Number of Hypotheses', 'y': 'Genus'}
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Clinical implications
        st.markdown("---")
        st.markdown("#### Clinical Implications")
        
        st.info("""
        **Key Findings Summary:**
        
        Based on the integrated analysis, the following microbial alterations are associated with asthma:
        
        1. **Gut-Lung Axis Dysbiosis**: Changes in butyrate-producing bacteria may affect systemic inflammation
        2. **Pathogen Enrichment**: Increased virulence factors from respiratory pathogens
        3. **Immune Dysregulation**: Alterations in immune-modulating taxa
        4. **Metabolic Shifts**: Changes in histamine and LPS production pathways
        
        These findings suggest potential therapeutic targets for microbiome-based interventions.
        """)
        
        # Generate report summary
        st.markdown("---")
        
        if st.button("📄 Generate Summary Report"):
            
            report_text = f"""
# AIsthma Forge Analysis Report

## Summary Statistics

- **Total Hypotheses Generated**: {len(hypotheses_df)}
- **High Priority Hypotheses**: {(hypotheses_df['priority'] == 1).sum()}
- **Unique Genera Implicated**: {hypotheses_df['genus'].nunique()}
- **Pathways Affected**: {hypotheses_df['pathway'].nunique()}

## Top 5 Gene Hypotheses

"""
            
            for idx, row in hypotheses_df.sort_values('priority').head(5).iterrows():
                report_text += f"""
### {idx + 1}. {row['genus']} - {row['pathway_description']}

**Hypothesis**: {row['hypothesis']}

**Priority**: {row['priority']} | **Direction**: {row['direction']}

---
"""
            
            report_text += """

## Pathway Summary

"""
            
            pathway_summary = hypotheses_df.groupby('pathway')['feature'].count().sort_values(ascending=False)
            
            for pathway, count in pathway_summary.items():
                report_text += f"- **{pathway.replace('_', ' ').title()}**: {count} features\n"
            
            report_text += """

## Recommendations

1. Validate top hypotheses with targeted functional metagenomics
2. Consider pathway-specific interventions (e.g., butyrate supplementation)
3. Investigate virulence factor expression in identified pathogens
4. Explore immune modulation through probiotic interventions

---

*Generated by AIsthma Forge - Microbiome Analysis Platform*
"""
            
            st.session_state.analysis_report = report_text
            
            st.success("✅ Report generated! View in 'Reports & Export' section")
            
            # Download report
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=report_text,
                file_name="aisthma_forge_report.md",
                mime="text/markdown"
            )
