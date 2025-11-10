# AIsthma Forge User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Step-by-Step Workflow](#step-by-step-workflow)
5. [Interpreting Results](#interpreting-results)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

## Introduction

AIsthma Forge is designed to make microbiome analysis accessible to researchers without extensive bioinformatics training. This guide walks you through the complete analysis workflow, from data upload to hypothesis generation.

### Who Should Use This Tool?

AIsthma Forge is ideal for researchers studying the relationship between microbiome composition and asthma, including clinicians, microbiologists, immunologists, and bioinformaticians seeking a streamlined analysis platform.

### What You Need

Before starting, ensure you have prepared OTU or ASV count tables from your sequencing data, along with sample metadata indicating asthma status and relevant covariates such as age, sex, and treatment history.

## Installation

### System Requirements

The application requires Python 3.11 or higher and runs on Linux, macOS, or Windows operating systems. A minimum of 4GB RAM is recommended, with 8GB or more preferred for larger datasets.

### Setup Instructions

Create a project directory and navigate into it. Set up a Python virtual environment to isolate dependencies. Install all required packages using the provided requirements file. Once installation completes, you can launch the application using the run script or directly via Streamlit.

## Data Preparation

### OTU/ASV Count Table Format

Your count table should be organized with features (OTUs or ASVs) as rows and samples as columns. The first column should contain feature identifiers, ideally with taxonomic annotations in the format `Kingdom|Phylum|Class|Order|Family|Genus|Species`. Values should represent raw read counts as integers.

### Metadata Format

The metadata file must include a `SampleID` column matching the sample names in your count table. An `Asthma_Status` column is required to distinguish between asthma and healthy control samples. Additional columns may include demographic variables, clinical measurements, or experimental conditions.

### Functional Data (Optional)

If you have performed functional prediction using tools like PICRUSt2 or HUMAnN, you can upload these results for integrated analysis. The format should match the count table structure with gene families or pathways as rows.

## Step-by-Step Workflow

### Step 1: Data Upload

Begin by navigating to the "Home & Data Upload" section in the sidebar. Upload your OTU count table and metadata file using the file upload widgets. The system will automatically detect the delimiter (comma or tab) and validate that sample IDs match between files. If you want to explore the platform without your own data, load the synthetic demo dataset to familiarize yourself with the interface.

### Step 2: Quality Control and Preprocessing

Move to the "Preprocessing" section to review data quality. Examine the sequencing depth distribution to identify samples with unusually low read counts. Check feature prevalence to understand how many samples contain each taxon. Apply filtering to remove low-quality samples and rare features that may represent sequencing artifacts. Set minimum sequencing depth based on the distribution, typically removing samples below the 10th percentile. For features, require presence in at least 10% of samples and a minimum total abundance across all samples.

### Step 3: Normalization

Choose an appropriate normalization method based on your downstream analysis goals. Total Sum Scaling (TSS) converts counts to relative abundances and is suitable for most diversity analyses. Rarefaction subsamples all libraries to equal depth, which is conservative but discards data. Centered Log-Ratio (CLR) transformation is recommended for compositional data analysis and machine learning. For differential abundance testing with DESeq2, retain raw counts without normalization.

### Step 4: Diversity Analysis

In the "Diversity Analysis" section, calculate alpha diversity metrics to assess within-sample microbial richness and evenness. Compare these metrics between asthma and healthy groups using appropriate statistical tests. The system automatically performs t-tests or Mann-Whitney U tests for two-group comparisons and ANOVA for multiple groups.

Proceed to beta diversity analysis to examine between-sample differences in community composition. Bray-Curtis dissimilarity is recommended for abundance-based comparisons, while Jaccard distance focuses on presence-absence patterns. Perform PERMANOVA testing to determine whether community composition differs significantly between groups. Visualize results using PCA or t-SNE ordination, coloring points by asthma status or other metadata variables.

### Step 5: Differential Abundance Testing

Navigate to "Differential Abundance" to identify specific taxa that differ between groups. Select your comparison groups (typically asthma versus healthy controls) and choose DESeq2 for count-based analysis with proper normalization and dispersion estimation. Set significance thresholds, commonly using adjusted p-value less than 0.05 and absolute log2 fold-change greater than 1.

Review the volcano plot to visualize effect sizes and statistical significance. Features in the upper corners represent strongly differentially abundant taxa. Examine the top features table to identify specific microbes enriched or depleted in asthma samples. Click on individual features to view their abundance distributions across groups.

### Step 6: Machine Learning Classification

In the "Machine Learning" section, build predictive models to classify samples based on their microbial profiles. Configure your model by selecting the target variable (typically asthma status) and choosing between Random Forest and XGBoost algorithms. Adjust hyperparameters such as the number of trees and maximum depth, or use default values for initial exploration.

Train the model and evaluate its performance using the test set. Review accuracy, precision, recall, F1 score, and ROC AUC metrics. High accuracy (above 0.80) suggests strong predictive signal in the microbiome data. Examine the confusion matrix to understand classification errors.

### Step 7: SHAP Explainability

After training your model, calculate SHAP values to understand which features drive predictions. The SHAP summary plot shows the most important features and how their values affect model output. Red points indicate high feature values pushing predictions toward the positive class, while blue points show low values. Use waterfall plots to explain individual sample predictions, revealing which specific taxa contribute to classifying a sample as asthma or healthy.

### Step 8: Generate Gene Hypotheses

Navigate to "Predictive Modeling" to integrate your differential abundance and machine learning results. The system automatically generates hypotheses linking identified taxa to known asthma-related pathways. These include virulence factors in pathogenic bacteria, butyrate production deficits in beneficial microbes, lipopolysaccharide biosynthesis driving inflammation, and immune modulation pathways.

Review the ranked hypotheses, prioritized by statistical significance and biological relevance. High-priority hypotheses represent strong candidates for experimental validation. Explore the pathway mapping section to understand the biological mechanisms underlying each hypothesis.

### Step 9: Export Results and Generate Reports

Finally, visit "Reports & Export" to download your analysis results. Export individual result tables as CSV files for further analysis or publication. Generate a comprehensive markdown report summarizing all analysis steps, key findings, and recommendations. This report can be converted to PDF for sharing with collaborators or including in grant applications.

## Interpreting Results

### Alpha Diversity

Lower alpha diversity in asthma samples suggests reduced microbial richness, which may indicate dysbiosis. Significant differences in Shannon or Simpson indices reflect changes in both richness and evenness of the community.

### Beta Diversity

Significant PERMANOVA results indicate that asthma and healthy samples have distinct microbial community structures. Ordination plots showing clear separation between groups visualize this difference. Overlapping clusters suggest similar microbiome composition despite different disease states.

### Differential Abundance

Features with positive log2 fold-change are enriched in the test group (typically asthma), while negative values indicate depletion. Large effect sizes (absolute log2FC greater than 2) represent substantial abundance differences. Consider both statistical significance and biological relevance when prioritizing features for follow-up.

### Machine Learning Performance

High model accuracy suggests the microbiome contains predictive information about asthma status. However, consider the balance between sensitivity (recall) and specificity when evaluating clinical utility. SHAP values reveal which specific taxa drive predictions, providing biological interpretability beyond simple feature importance.

### Gene Hypotheses

High-priority hypotheses integrate statistical evidence with known biological pathways. These represent testable predictions about functional gene changes in the microbiome. Hypotheses involving virulence factors suggest pathogen-driven inflammation, while those related to butyrate production indicate loss of protective metabolic functions.

## Troubleshooting

### Data Upload Issues

If sample IDs do not match between count table and metadata, verify that both files use consistent naming conventions. Check for extra spaces, special characters, or case sensitivity issues. Ensure the metadata file includes a column explicitly named `SampleID`.

### Preprocessing Errors

If normalization fails, verify that your count table contains numeric values without missing data. For DESeq2 analysis, ensure counts are integers rather than decimal values. If rarefaction fails, check that all samples have sufficient sequencing depth.

### Analysis Failures

DESeq2 may fail with very small sample sizes (fewer than 3 samples per group) or when all samples have identical counts for a feature. In such cases, use the alternative Mann-Whitney U test method. Machine learning requires sufficient samples for train-test splitting, typically at least 20 samples total.

### Performance Issues

Large datasets (thousands of features across hundreds of samples) may require several minutes for analysis. Consider applying more stringent filtering to reduce feature count. SHAP calculation is computationally intensive and may take longer for complex models.

## Best Practices

### Study Design

Include sufficient sample sizes for robust statistical analysis, ideally at least 20 samples per group. Control for confounding variables such as age, sex, antibiotic use, and geographic location. Consider longitudinal sampling to track microbiome changes over time.

### Data Quality

Perform thorough quality control before analysis, removing samples with very low sequencing depth or unusual characteristics. Filter rare features that appear in only a few samples, as these may represent sequencing errors or contamination. Document all filtering and normalization decisions for reproducibility.

### Statistical Considerations

Apply appropriate multiple testing correction (FDR) when testing many features simultaneously. Use effect size thresholds in addition to p-values to identify biologically meaningful differences. Validate findings in independent cohorts when possible.

### Biological Interpretation

Consider the ecological context of identified taxa, including their known metabolic capabilities and interactions with the host immune system. Integrate findings with existing literature on asthma pathogenesis and microbiome function. Prioritize hypotheses that align with established biological mechanisms while remaining open to novel discoveries.

### Reproducibility

Document all analysis parameters and software versions. Export and archive raw results along with processed data. Share analysis code and workflows to enable reproduction by other researchers. Consider depositing data in public repositories such as NCBI SRA or EBI ENA.

---

For additional support, consult the README file or open an issue on the project GitHub repository. The AIsthma Forge community welcomes questions, suggestions, and contributions to improve the platform for all users.
