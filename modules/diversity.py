"""
Diversity Analysis Module
Alpha and beta diversity metrics with statistical testing
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

def calculate_alpha_diversity(otu_table):
    """Calculate alpha diversity metrics"""
    
    metrics = {}
    
    for sample in otu_table.columns:
        counts = otu_table[sample].values
        counts = counts[counts > 0]  # Remove zeros
        
        # Observed richness
        richness = len(counts)
        
        # Shannon diversity
        proportions = counts / counts.sum()
        shannon = -np.sum(proportions * np.log(proportions))
        
        # Simpson diversity
        simpson = 1 - np.sum(proportions ** 2)
        
        # Pielou's evenness
        evenness = shannon / np.log(richness) if richness > 1 else 0
        
        metrics[sample] = {
            'Richness': richness,
            'Shannon': shannon,
            'Simpson': simpson,
            'Evenness': evenness
        }
    
    return pd.DataFrame(metrics).T

def calculate_beta_diversity(otu_table, method='bray_curtis'):
    """Calculate beta diversity distance matrix"""
    
    # Transpose so samples are rows
    data = otu_table.T.values
    
    if method == 'bray_curtis':
        # Bray-Curtis dissimilarity
        distances = []
        for i in range(len(data)):
            row_distances = []
            for j in range(len(data)):
                numerator = np.sum(np.abs(data[i] - data[j]))
                denominator = np.sum(data[i] + data[j])
                if denominator == 0:
                    distance = 0
                else:
                    distance = numerator / denominator
                row_distances.append(distance)
            distances.append(row_distances)
        dist_matrix = np.array(distances)
    
    elif method == 'jaccard':
        # Jaccard distance (presence/absence)
        binary_data = (data > 0).astype(int)
        dist_matrix = squareform(pdist(binary_data, metric='jaccard'))
    
    elif method == 'euclidean':
        dist_matrix = squareform(pdist(data, metric='euclidean'))
    
    else:
        dist_matrix = squareform(pdist(data, metric='euclidean'))
    
    return pd.DataFrame(dist_matrix, index=otu_table.columns, columns=otu_table.columns)

def perform_permanova(distance_matrix, metadata, group_column, n_permutations=999):
    """Perform PERMANOVA test"""
    
    # Simplified PERMANOVA implementation
    samples = distance_matrix.index
    groups = metadata.set_index('SampleID' if 'SampleID' in metadata.columns else metadata.columns[0])[group_column]
    groups = groups.loc[samples]
    
    # Calculate F-statistic
    unique_groups = groups.unique()
    n_samples = len(samples)
    
    # Total sum of squares
    dist_values = distance_matrix.values
    ss_total = np.sum(dist_values ** 2) / n_samples
    
    # Within-group sum of squares
    ss_within = 0
    for group in unique_groups:
        group_samples = groups[groups == group].index
        group_dist = distance_matrix.loc[group_samples, group_samples].values
        ss_within += np.sum(group_dist ** 2) / len(group_samples)
    
    # Between-group sum of squares
    ss_between = ss_total - ss_within
    
    # Degrees of freedom
    df_between = len(unique_groups) - 1
    df_within = n_samples - len(unique_groups)
    
    # F-statistic
    f_stat = (ss_between / df_between) / (ss_within / df_within) if df_within > 0 else 0
    
    # Permutation test
    perm_f_stats = []
    for _ in range(n_permutations):
        perm_groups = groups.sample(frac=1).values
        perm_groups = pd.Series(perm_groups, index=groups.index)
        
        perm_ss_within = 0
        for group in unique_groups:
            group_samples = perm_groups[perm_groups == group].index
            if len(group_samples) > 0:
                group_dist = distance_matrix.loc[group_samples, group_samples].values
                perm_ss_within += np.sum(group_dist ** 2) / len(group_samples)
        
        perm_ss_between = ss_total - perm_ss_within
        perm_f = (perm_ss_between / df_between) / (perm_ss_within / df_within) if df_within > 0 else 0
        perm_f_stats.append(perm_f)
    
    # Calculate p-value
    p_value = np.sum(np.array(perm_f_stats) >= f_stat) / n_permutations
    
    return {
        'F_statistic': f_stat,
        'p_value': p_value,
        'R_squared': ss_between / ss_total
    }

def render_diversity_page():
    """Render the diversity analysis page"""
    
    st.header("📈 Diversity Analysis")
    
    if 'otu_table_final' not in st.session_state:
        st.warning("⚠️ Please complete preprocessing first.")
        return
    
    otu_table = st.session_state.otu_table_final.copy()
    metadata = st.session_state.metadata_filtered if 'metadata_filtered' in st.session_state else st.session_state.metadata
    
    # Ensure we're working with matching samples
    common_samples = list(set(otu_table.columns) & set(metadata['SampleID'] if 'SampleID' in metadata.columns else metadata.index))
    otu_table = otu_table[common_samples]
    
    st.markdown("""
    Diversity metrics quantify the richness and evenness of microbial communities.
    Compare diversity between asthma and healthy groups to identify dysbiosis patterns.
    """)
    
    tab1, tab2, tab3 = st.tabs([
        "🔢 Alpha Diversity",
        "🌐 Beta Diversity",
        "📊 Ordination"
    ])
    
    with tab1:
        st.subheader("Alpha Diversity Analysis")
        
        st.markdown("""
        **Alpha diversity** measures within-sample diversity:
        - **Richness**: Number of unique features
        - **Shannon**: Accounts for abundance and evenness
        - **Simpson**: Probability two random reads are different taxa
        - **Evenness**: How evenly distributed abundances are
        """)
        
        # Calculate alpha diversity
        with st.spinner("Calculating alpha diversity metrics..."):
            alpha_div = calculate_alpha_diversity(otu_table)
        
        # Merge with metadata
        if 'SampleID' in metadata.columns:
            metadata_indexed = metadata.set_index('SampleID')
        else:
            metadata_indexed = metadata
        
        alpha_div = alpha_div.merge(metadata_indexed, left_index=True, right_index=True, how='left')
        
        # Store in session state
        st.session_state.alpha_diversity = alpha_div
        
        # Select metric to visualize
        metric = st.selectbox(
            "Select Alpha Diversity Metric:",
            ['Richness', 'Shannon', 'Simpson', 'Evenness']
        )
        
        # Select grouping variable
        categorical_cols = metadata.select_dtypes(include=['object', 'category']).columns.tolist()
        if 'SampleID' in categorical_cols:
            categorical_cols.remove('SampleID')
        
        if len(categorical_cols) > 0:
            group_by = st.selectbox("Group by:", categorical_cols, index=0)
            
            # Create visualization
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Box plot
                fig = px.box(
                    alpha_div,
                    x=group_by,
                    y=metric,
                    color=group_by,
                    points='all',
                    title=f"{metric} by {group_by}"
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Statistical Test")
                
                # Perform statistical test
                groups = alpha_div[group_by].unique()
                
                if len(groups) == 2:
                    group1_data = alpha_div[alpha_div[group_by] == groups[0]][metric]
                    group2_data = alpha_div[alpha_div[group_by] == groups[1]][metric]
                    
                    # T-test
                    t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
                    
                    # Mann-Whitney U test
                    u_stat, p_value_mw = stats.mannwhitneyu(group1_data, group2_data)
                    
                    st.write(f"**{groups[0]}** (n={len(group1_data)})")
                    st.write(f"Mean: {group1_data.mean():.3f}")
                    st.write(f"Median: {group1_data.median():.3f}")
                    
                    st.write(f"**{groups[1]}** (n={len(group2_data)})")
                    st.write(f"Mean: {group2_data.mean():.3f}")
                    st.write(f"Median: {group2_data.median():.3f}")
                    
                    st.markdown("---")
                    st.write("**T-test:**")
                    st.write(f"p-value = {p_value:.4f}")
                    
                    st.write("**Mann-Whitney U:**")
                    st.write(f"p-value = {p_value_mw:.4f}")
                    
                    if p_value < 0.05:
                        st.success("✅ Significant difference (p < 0.05)")
                    else:
                        st.info("No significant difference (p ≥ 0.05)")
                
                elif len(groups) > 2:
                    # ANOVA
                    group_data = [alpha_div[alpha_div[group_by] == g][metric].values for g in groups]
                    f_stat, p_value = stats.f_oneway(*group_data)
                    
                    st.write("**ANOVA:**")
                    st.write(f"F-statistic = {f_stat:.3f}")
                    st.write(f"p-value = {p_value:.4f}")
                    
                    if p_value < 0.05:
                        st.success("✅ Significant difference (p < 0.05)")
                    else:
                        st.info("No significant difference (p ≥ 0.05)")
            
            # Summary table
            st.markdown("---")
            st.markdown("#### Summary Statistics by Group")
            
            summary = alpha_div.groupby(group_by)[['Richness', 'Shannon', 'Simpson', 'Evenness']].agg(['mean', 'std', 'median'])
            st.dataframe(summary.round(3))
        
        else:
            st.warning("No categorical variables found in metadata for grouping.")
            st.dataframe(alpha_div)
    
    with tab2:
        st.subheader("Beta Diversity Analysis")
        
        st.markdown("""
        **Beta diversity** measures between-sample diversity (dissimilarity).
        Lower values indicate more similar communities.
        """)
        
        # Select distance metric
        distance_method = st.selectbox(
            "Select Distance Metric:",
            ['bray_curtis', 'jaccard', 'euclidean'],
            format_func=lambda x: {
                'bray_curtis': 'Bray-Curtis (abundance-based)',
                'jaccard': 'Jaccard (presence/absence)',
                'euclidean': 'Euclidean'
            }[x]
        )
        
        # Calculate beta diversity
        with st.spinner("Calculating beta diversity..."):
            # For beta diversity, use relative abundance or appropriate transformation
            if otu_table.sum(axis=0).mean() > 10:  # If not already normalized
                otu_for_beta = otu_table.div(otu_table.sum(axis=0), axis=1)
            else:
                otu_for_beta = otu_table
            
            beta_div = calculate_beta_diversity(otu_for_beta, method=distance_method)
        
        st.session_state.beta_diversity = beta_div
        
        # Heatmap
        st.markdown("#### Distance Matrix Heatmap")
        
        fig = px.imshow(
            beta_div.values,
            x=beta_div.columns,
            y=beta_div.index,
            color_continuous_scale='Viridis',
            title=f"{distance_method.replace('_', ' ').title()} Distance Matrix"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # PERMANOVA test
        if len(categorical_cols) > 0:
            st.markdown("---")
            st.markdown("#### PERMANOVA Test")
            
            st.info("""
            **PERMANOVA** (Permutational Multivariate Analysis of Variance) tests whether 
            microbial community composition differs significantly between groups.
            """)
            
            permanova_group = st.selectbox("Test group differences:", categorical_cols, key='permanova_group')
            n_permutations = st.slider("Number of permutations:", 99, 9999, 999)
            
            if st.button("Run PERMANOVA", type="primary"):
                with st.spinner("Running permutation test..."):
                    permanova_results = perform_permanova(
                        beta_div,
                        metadata,
                        permanova_group,
                        n_permutations=n_permutations
                    )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("F-statistic", f"{permanova_results['F_statistic']:.4f}")
                
                with col2:
                    st.metric("R²", f"{permanova_results['R_squared']:.4f}")
                
                with col3:
                    st.metric("p-value", f"{permanova_results['p_value']:.4f}")
                
                if permanova_results['p_value'] < 0.05:
                    st.success(f"✅ Significant difference in community composition between {permanova_group} groups (p < 0.05)")
                else:
                    st.info(f"No significant difference in community composition between {permanova_group} groups (p ≥ 0.05)")
    
    with tab3:
        st.subheader("Ordination Analysis")
        
        st.markdown("""
        **Ordination** visualizes beta diversity in 2D/3D space.
        Samples that cluster together have similar microbial communities.
        """)
        
        ordination_method = st.selectbox(
            "Select Ordination Method:",
            ['PCA', 't-SNE'],
            format_func=lambda x: {
                'PCA': 'PCA (Principal Component Analysis)',
                't-SNE': 't-SNE (t-Distributed Stochastic Neighbor Embedding)'
            }[x]
        )
        
        # Prepare data
        if otu_table.sum(axis=0).mean() > 10:
            otu_for_ord = otu_table.div(otu_table.sum(axis=0), axis=1).T
        else:
            otu_for_ord = otu_table.T
        
        # Replace any NaN or inf values
        otu_for_ord = otu_for_ord.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        if ordination_method == 'PCA':
            with st.spinner("Running PCA..."):
                pca = PCA(n_components=min(3, len(otu_for_ord)))
                coords = pca.fit_transform(otu_for_ord.values)
                
                explained_var = pca.explained_variance_ratio_
                
                ord_df = pd.DataFrame({
                    'PC1': coords[:, 0],
                    'PC2': coords[:, 1],
                    'SampleID': otu_for_ord.index
                })
                
                if coords.shape[1] >= 3:
                    ord_df['PC3'] = coords[:, 2]
                
                st.info(f"PC1 explains {explained_var[0]*100:.1f}% of variance, PC2 explains {explained_var[1]*100:.1f}%")
        
        else:  # t-SNE
            perplexity = st.slider("Perplexity:", 5, 50, 30)
            
            with st.spinner("Running t-SNE (this may take a moment)..."):
                tsne = TSNE(n_components=2, perplexity=min(perplexity, len(otu_for_ord)-1), random_state=42)
                coords = tsne.fit_transform(otu_for_ord.values)
                
                ord_df = pd.DataFrame({
                    'tSNE1': coords[:, 0],
                    'tSNE2': coords[:, 1],
                    'SampleID': otu_for_ord.index
                })
        
        # Merge with metadata
        if 'SampleID' in metadata.columns:
            ord_df = ord_df.merge(metadata, on='SampleID', how='left')
        else:
            metadata_with_id = metadata.copy()
            metadata_with_id['SampleID'] = metadata_with_id.index
            ord_df = ord_df.merge(metadata_with_id, on='SampleID', how='left')
        
        # Visualization
        if len(categorical_cols) > 0:
            color_by = st.selectbox("Color by:", categorical_cols, key='ord_color')
            
            if ordination_method == 'PCA':
                fig = px.scatter(
                    ord_df,
                    x='PC1',
                    y='PC2',
                    color=color_by,
                    hover_data=['SampleID'],
                    title=f"PCA Ordination colored by {color_by}",
                    labels={'PC1': f'PC1 ({explained_var[0]*100:.1f}%)', 
                           'PC2': f'PC2 ({explained_var[1]*100:.1f}%)'}
                )
            else:
                fig = px.scatter(
                    ord_df,
                    x='tSNE1',
                    y='tSNE2',
                    color=color_by,
                    hover_data=['SampleID'],
                    title=f"t-SNE Ordination colored by {color_by}"
                )
            
            fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("No categorical variables for coloring.")
        
        # Download coordinates
        st.markdown("---")
        csv = ord_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Ordination Coordinates",
            data=csv,
            file_name=f"{ordination_method}_coordinates.csv",
            mime="text/csv"
        )
