# AIsthma Forge - Project Summary

## Overview

**AIsthma Forge** is a comprehensive, open-source web application designed to transform raw microbiome sequencing data into actionable insights for asthma research. Built with Python and Streamlit, it provides an end-to-end analysis pipeline accessible to researchers without extensive bioinformatics expertise.

## Key Features

### 1. Data Management
- Support for multiple input formats (FASTQ, OTU, ASV counts)
- Metadata integration with validation
- Optional functional prediction data (PICRUSt2/HUMAnN)
- Built-in synthetic dataset for demonstration

### 2. Preprocessing Pipeline
- Quality control metrics and visualization
- Flexible filtering options (sample depth, feature prevalence)
- Multiple normalization methods (TSS, Rarefaction, CLR, CSS, TMM)
- Variance-stabilizing transformations

### 3. Diversity Analysis
- **Alpha diversity**: Richness, Shannon, Simpson, Evenness indices
- **Beta diversity**: Bray-Curtis, Jaccard, Euclidean distances
- Statistical testing (t-test, Mann-Whitney U, ANOVA, PERMANOVA)
- Ordination visualization (PCA, t-SNE)

### 4. Differential Abundance
- DESeq2-like methodology via pydeseq2
- Alternative Mann-Whitney U testing
- Multiple testing correction (FDR)
- Interactive volcano plots and feature exploration

### 5. Machine Learning
- Random Forest and XGBoost classifiers
- Cross-validation and comprehensive performance metrics
- Feature importance analysis
- **SHAP explainability** for model interpretation
- Single-sample prediction for clinical risk scoring

### 6. Predictive Modeling
- Automated gene hypothesis generation
- Integration of differential abundance and ML results
- Pathway mapping to known asthma-related functions:
  - Virulence factors (Moraxella, Haemophilus)
  - Butyrate production (Faecalibacterium, Roseburia)
  - LPS biosynthesis (pro-inflammatory)
  - Histamine metabolism (allergic response)
  - Immune modulation (Bacteroides, Bifidobacterium)
- Ranked hypotheses for experimental validation

### 7. Reporting and Export
- Comprehensive analysis summaries
- Multiple export formats (CSV, JSON, Markdown)
- Automated report generation
- Downloadable results for all analysis steps

## Technical Architecture

### Technology Stack
- **Framework**: Streamlit (Python web framework)
- **Data Processing**: Pandas, NumPy
- **Statistics**: SciPy, scikit-bio
- **Differential Analysis**: pydeseq2
- **Machine Learning**: scikit-learn, XGBoost
- **Explainability**: SHAP
- **Visualization**: Plotly, Matplotlib, Seaborn

### Project Structure
```
aisthma_forge/
├── app.py                          # Main application entry point
├── modules/                        # Analysis modules
│   ├── __init__.py
│   ├── data_upload.py             # Data import and validation
│   ├── preprocessing.py           # QC and normalization
│   ├── diversity.py               # Alpha/beta diversity
│   ├── differential_abundance.py  # DESeq2 analysis
│   ├── machine_learning.py        # ML classification + SHAP
│   ├── predictive_modeling.py     # Gene hypothesis generation
│   └── reports.py                 # Report generation
├── .streamlit/
│   └── config.toml                # Application configuration
├── data/                          # Data directory
├── assets/                        # Static assets
├── reports/                       # Generated reports
├── requirements.txt               # Python dependencies
├── run.sh                         # Launch script
├── README.md                      # Project documentation
├── USER_GUIDE.md                  # Detailed user guide
├── DEPLOYMENT.md                  # Deployment instructions
└── LICENSE                        # MIT License
```

## Scientific Background

Asthma affects over 300 million people globally, with microbiome dysbiosis emerging as a key driver of inflammation through the gut-lung axis. Traditional microbiome analysis tools are fragmented and require command-line expertise. AIsthma Forge addresses this gap by providing an integrated, user-friendly platform that combines statistical rigor with machine learning explainability.

### Key Biological Pathways

The application focuses on five major pathway categories linked to asthma pathogenesis:

1. **Virulence Factors**: Bacterial components (LPS, peptidoglycan, flagella) that trigger inflammatory responses
2. **Butyrate Production**: Short-chain fatty acid synthesis with anti-inflammatory properties
3. **LPS Biosynthesis**: Pro-inflammatory lipopolysaccharide production pathways
4. **Histamine Metabolism**: Modulation of allergic responses through histamine production/degradation
5. **Immune Modulation**: Polysaccharide-mediated regulation of host immune responses

## Usage Workflow

### Typical Analysis Pipeline

1. **Upload Data**: Import OTU/ASV counts and metadata
2. **Preprocess**: Apply QC filters and normalization
3. **Explore Diversity**: Calculate alpha/beta diversity metrics
4. **Identify Markers**: Run differential abundance testing
5. **Build Models**: Train ML classifiers with SHAP interpretation
6. **Generate Hypotheses**: Link findings to biological pathways
7. **Export Results**: Download comprehensive reports

### Time Requirements

- Small dataset (<50 samples, <500 features): 5-10 minutes
- Medium dataset (50-200 samples, 500-2000 features): 15-30 minutes
- Large dataset (>200 samples, >2000 features): 30-60 minutes

## Validation and Testing

### Test Data

The application includes a synthetic dataset generator that creates realistic microbiome data for demonstration purposes. This allows users to explore all features without requiring their own data.

### Quality Assurance

- Input validation for data formats
- Error handling throughout pipeline
- Statistical method validation against published tools
- Cross-validation for ML models

## Deployment Options

### Local Deployment
- Simple script-based launch (`./run.sh`)
- Runs on localhost:8501
- Suitable for individual researchers

### Cloud Deployment
- **Streamlit Community Cloud**: Free hosting for public apps
- **Docker**: Containerized deployment for any platform
- **AWS/GCP/Azure**: Enterprise-grade cloud hosting
- **Kubernetes**: Scalable multi-user deployment

## Performance Characteristics

### Computational Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores
- **Large datasets**: 16GB+ RAM, 8+ CPU cores

### Scalability
- Handles datasets up to 1000 samples × 5000 features on standard hardware
- Larger datasets require increased memory allocation
- ML training scales linearly with sample size
- SHAP calculation is most computationally intensive step

## Future Enhancements

### Planned Features
- Integration with public microbiome databases (QIITA, MGnify)
- Support for metagenomic shotgun data
- Longitudinal analysis capabilities
- Multi-omics integration (metabolomics, transcriptomics)
- Batch processing for multiple datasets
- User authentication and project management
- Persistent storage for analysis history

### Research Applications
- Asthma endotyping and biomarker discovery
- Treatment response prediction
- Microbiome-based therapeutic target identification
- Gut-lung axis mechanistic studies
- Clinical risk stratification

## Impact and Applications

### Research Benefits
- **Accessibility**: No bioinformatics expertise required
- **Integration**: Combines multiple analysis approaches
- **Reproducibility**: Standardized pipeline with documented parameters
- **Interpretability**: SHAP explainability for ML models
- **Hypothesis Generation**: Automated linking to biological pathways

### Clinical Potential
- Risk stratification for asthma development
- Personalized treatment selection
- Monitoring of microbiome-based interventions
- Identification of therapeutic targets

### Educational Value
- Teaching tool for microbiome analysis concepts
- Demonstration of ML in biological research
- Example of reproducible computational workflows

## Acknowledgments

AIsthma Forge builds upon the work of numerous open-source projects and scientific communities:

- **Streamlit**: Web framework enabling rapid application development
- **DESeq2**: Statistical methodology for differential abundance
- **SHAP**: Explainable AI framework
- **scikit-learn/XGBoost**: Machine learning algorithms
- **Microbiome research community**: Scientific foundation and validation

## Citation

If you use AIsthma Forge in your research, please cite:

```
AIsthma Forge: An Open-Source Web Application for Microbiome Analysis in Asthma Research
[Authors], 2025
GitHub: https://github.com/yourusername/aisthma-forge
```

## License

AIsthma Forge is released under the MIT License, allowing free use, modification, and distribution with attribution.

## Contact and Support

- **GitHub Repository**: [https://github.com/yourusername/aisthma-forge]
- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Join the community on GitHub Discussions
- **Email**: [your.email@institution.edu]

---

## Quick Start Commands

```bash
# Clone repository
git clone https://github.com/yourusername/aisthma-forge.git
cd aisthma-forge

# Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Launch application
./run.sh
# or
streamlit run app.py
```

## Statistics

- **Lines of Code**: ~3,500+ Python
- **Modules**: 7 analysis modules
- **Features**: 50+ analysis functions
- **Visualizations**: 20+ interactive plots
- **Documentation**: 4 comprehensive guides
- **Dependencies**: 16 core packages

---

**Version**: 1.0.0  
**Release Date**: November 2025  
**Status**: Production Ready  
**Maintainer**: AIsthma Forge Development Team

🫁 **Accelerating asthma research through accessible, integrated microbiome analysis**
