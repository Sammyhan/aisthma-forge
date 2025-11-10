#!/bin/bash

# AIsthma Forge Deployment Script

echo "🫁 Starting AIsthma Forge..."

# Activate virtual environment
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Run Streamlit app
echo "🚀 Launching application at http://localhost:8501"
streamlit run app.py
