"""
Machine Learning Module
Classification with Random Forest/XGBoost and SHAP explainability
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

def prepare_ml_data(otu_table, metadata, target_column):
    """Prepare data for machine learning"""
    
    # Transpose so samples are rows
    X = otu_table.T
    
    # Get target variable
    if 'SampleID' in metadata.columns:
        metadata = metadata.set_index('SampleID')
    
    # Align samples
    common_samples = list(set(X.index) & set(metadata.index))
    X = X.loc[common_samples]
    y = metadata.loc[common_samples, target_column]
    
    return X, y

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """Train Random Forest classifier"""
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    return clf

def train_xgboost(X_train, y_train, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
    """Train XGBoost classifier"""
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    clf.fit(X_train, y_train_encoded)
    
    # Store label encoder
    clf.label_encoder = le
    
    return clf

def evaluate_model(clf, X_test, y_test):
    """Evaluate model performance"""
    
    # Handle XGBoost label encoding
    if hasattr(clf, 'label_encoder'):
        y_test_encoded = clf.label_encoder.transform(y_test)
        y_pred_encoded = clf.predict(X_test)
        y_pred = clf.label_encoder.inverse_transform(y_pred_encoded)
        y_pred_proba = clf.predict_proba(X_test)
    else:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Handle binary vs multiclass
    average_method = 'binary' if len(np.unique(y_test)) == 2 else 'weighted'
    
    precision = precision_score(y_test, y_pred, average=average_method, zero_division=0)
    recall = recall_score(y_test, y_pred, average=average_method, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average_method, zero_division=0)
    
    # ROC AUC for binary classification
    if len(np.unique(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    return metrics, y_pred, y_pred_proba

def render_ml_page():
    """Render the machine learning page"""
    
    st.header("🤖 Machine Learning Classification")
    
    # Check for required data
    if 'otu_table_final' not in st.session_state:
        st.warning("⚠️ Please complete preprocessing first.")
        return
    
    otu_table = st.session_state.otu_table_final.copy()
    metadata = st.session_state.metadata_filtered if 'metadata_filtered' in st.session_state else st.session_state.metadata
    
    st.markdown("""
    Build machine learning models to classify samples based on microbial profiles.
    Use **SHAP** (SHapley Additive exPlanations) to identify key microbial features 
    driving predictions and understand model decisions.
    """)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️ Model Configuration",
        "📊 Model Performance",
        "🔍 SHAP Analysis",
        "🎯 Single Sample Prediction"
    ])
    
    with tab1:
        st.subheader("Configure Machine Learning Model")
        
        # Select target variable
        categorical_cols = metadata.select_dtypes(include=['object', 'category']).columns.tolist()
        if 'SampleID' in categorical_cols:
            categorical_cols.remove('SampleID')
        
        if len(categorical_cols) == 0:
            st.error("No categorical variables found for classification.")
            return
        
        target_column = st.selectbox(
            "Select target variable (what to predict):",
            categorical_cols,
            help="The outcome variable for classification (e.g., Asthma_Status)"
        )
        
        # Check class distribution
        if 'SampleID' in metadata.columns:
            target_dist = metadata.set_index('SampleID')[target_column].value_counts()
        else:
            target_dist = metadata[target_column].value_counts()
        
        st.write("**Class distribution:**")
        for class_name, count in target_dist.items():
            st.write(f"- {class_name}: {count} samples")
        
        if len(target_dist) < 2:
            st.error("Need at least 2 classes for classification.")
            return
        
        st.markdown("---")
        
        # Model selection
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select model type:",
                ["Random Forest", "XGBoost"],
                help="Both are ensemble methods suitable for microbiome data"
            )
        
        with col2:
            test_size = st.slider(
                "Test set size (%):",
                min_value=10,
                max_value=40,
                value=20,
                step=5
            ) / 100
        
        # Model hyperparameters
        st.markdown("#### Hyperparameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_estimators = st.number_input(
                "Number of trees:",
                min_value=10,
                max_value=500,
                value=100,
                step=10
            )
        
        with col2:
            max_depth = st.number_input(
                "Max depth (0=unlimited):",
                min_value=0,
                max_value=50,
                value=10,
                step=1
            )
            max_depth = None if max_depth == 0 else max_depth
        
        with col3:
            if model_type == "XGBoost":
                learning_rate = st.number_input(
                    "Learning rate:",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.1,
                    step=0.01
                )
        
        # Feature selection
        st.markdown("---")
        st.markdown("#### Feature Selection")
        
        feature_selection = st.checkbox("Use feature selection", value=False)
        
        if feature_selection:
            n_features = st.slider(
                "Number of top features to use:",
                min_value=10,
                max_value=min(500, len(otu_table)),
                value=min(100, len(otu_table)),
                step=10
            )
        else:
            n_features = len(otu_table)
        
        # Cross-validation
        use_cv = st.checkbox("Perform cross-validation", value=True)
        if use_cv:
            cv_folds = st.slider("Number of CV folds:", 3, 10, 5)
        
        # Train model
        if st.button("🚀 Train Model", type="primary"):
            
            with st.spinner("Training model..."):
                
                # Prepare data
                X, y = prepare_ml_data(otu_table, metadata, target_column)
                
                # Feature selection based on variance
                if feature_selection:
                    feature_var = X.var(axis=0).sort_values(ascending=False)
                    top_features = feature_var.head(n_features).index
                    X = X[top_features]
                    st.info(f"Using top {n_features} features based on variance")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Convert back to DataFrame for compatibility
                X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
                X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
                
                # Train model
                if model_type == "Random Forest":
                    clf = train_random_forest(X_train_scaled, y_train, n_estimators, max_depth)
                else:
                    clf = train_xgboost(X_train_scaled, y_train, n_estimators, max_depth, learning_rate)
                
                # Cross-validation
                if use_cv:
                    cv_scores = cross_val_score(
                        clf, X_train_scaled, y_train,
                        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                        scoring='accuracy'
                    )
                    st.session_state.cv_scores = cv_scores
                
                # Evaluate on test set
                metrics, y_pred, y_pred_proba = evaluate_model(clf, X_test_scaled, y_test)
                
                # Store everything in session state
                st.session_state.ml_model = clf
                st.session_state.ml_scaler = scaler
                st.session_state.ml_X_train = X_train_scaled
                st.session_state.ml_X_test = X_test_scaled
                st.session_state.ml_y_train = y_train
                st.session_state.ml_y_test = y_test
                st.session_state.ml_y_pred = y_pred
                st.session_state.ml_y_pred_proba = y_pred_proba
                st.session_state.ml_metrics = metrics
                st.session_state.ml_target = target_column
                st.session_state.ml_model_type = model_type
                
                st.success("✅ Model trained successfully!")
                st.info("Navigate to 'Model Performance' tab to view results")
    
    with tab2:
        st.subheader("Model Performance Metrics")
        
        if 'ml_model' not in st.session_state:
            st.info("👈 Please train a model first in the 'Model Configuration' tab")
            return
        
        metrics = st.session_state.ml_metrics
        y_test = st.session_state.ml_y_test
        y_pred = st.session_state.ml_y_pred
        y_pred_proba = st.session_state.ml_y_pred_proba
        
        # Display metrics
        st.markdown("#### Performance Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        
        with col2:
            st.metric("Precision", f"{metrics['precision']:.3f}")
        
        with col3:
            st.metric("Recall", f"{metrics['recall']:.3f}")
        
        with col4:
            st.metric("F1 Score", f"{metrics['f1']:.3f}")
        
        with col5:
            st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        
        # Cross-validation scores
        if 'cv_scores' in st.session_state:
            st.markdown("---")
            st.markdown("#### Cross-Validation Results")
            
            cv_scores = st.session_state.cv_scores
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean CV Accuracy", f"{cv_scores.mean():.3f}")
            with col2:
                st.metric("Std Dev", f"{cv_scores.std():.3f}")
            with col3:
                st.metric("Min-Max", f"{cv_scores.min():.3f} - {cv_scores.max():.3f}")
        
        # Confusion matrix
        st.markdown("---")
        st.markdown("#### Confusion Matrix")
        
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(y_test.unique())
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale='Blues'
        )
        fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # ROC curve for binary classification
        if len(labels) == 2:
            st.markdown("---")
            st.markdown("#### ROC Curve")
            
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1], pos_label=labels[1])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random classifier',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title='Receiver Operating Characteristic (ROC) Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=700,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.markdown("---")
        st.markdown("#### Feature Importance (Top 20)")
        
        clf = st.session_state.ml_model
        
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            feature_names = st.session_state.ml_X_train.columns
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(20)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 20 Most Important Features'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Download feature importance
            csv = importance_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Feature Importance",
                data=csv,
                file_name="feature_importance.csv",
                mime="text/csv"
            )
    
    with tab3:
        st.subheader("SHAP Analysis - Model Explainability")
        
        if 'ml_model' not in st.session_state:
            st.info("👈 Please train a model first")
            return
        
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** values explain individual predictions
        by showing how each feature contributes to the model's output.
        
        - **Positive SHAP value**: Feature pushes prediction toward positive class
        - **Negative SHAP value**: Feature pushes prediction toward negative class
        - **Magnitude**: Strength of the feature's impact
        """)
        
        clf = st.session_state.ml_model
        X_test = st.session_state.ml_X_test
        
        # Calculate SHAP values
        if st.button("🔍 Calculate SHAP Values", type="primary"):
            
            with st.spinner("Calculating SHAP values (this may take a minute)..."):
                
                # Create SHAP explainer
                if st.session_state.ml_model_type == "Random Forest":
                    explainer = shap.TreeExplainer(clf)
                else:
                    explainer = shap.TreeExplainer(clf)
                
                # Calculate SHAP values for test set
                shap_values = explainer.shap_values(X_test)
                
                # Store in session state
                st.session_state.shap_values = shap_values
                st.session_state.shap_explainer = explainer
                
                st.success("✅ SHAP values calculated!")
        
        if 'shap_values' in st.session_state:
            
            shap_values = st.session_state.shap_values
            
            # For binary classification, use positive class
            if isinstance(shap_values, list):
                shap_values_plot = shap_values[1]
            else:
                shap_values_plot = shap_values
            
            # Summary plot
            st.markdown("---")
            st.markdown("#### SHAP Summary Plot")
            
            st.info("This plot shows the most important features and their impact on predictions")
            
            # Create SHAP summary plot
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values_plot, X_test, show=False, max_display=20)
            st.pyplot(fig)
            plt.close()
            
            # Mean absolute SHAP values (feature importance)
            st.markdown("---")
            st.markdown("#### Mean Absolute SHAP Values")
            
            mean_shap = np.abs(shap_values_plot).mean(axis=0)
            shap_importance_df = pd.DataFrame({
                'Feature': X_test.columns,
                'Mean_SHAP': mean_shap
            }).sort_values('Mean_SHAP', ascending=False).head(20)
            
            fig = px.bar(
                shap_importance_df,
                x='Mean_SHAP',
                y='Feature',
                orientation='h',
                title='Top 20 Features by Mean Absolute SHAP Value'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Waterfall plot for individual prediction
            st.markdown("---")
            st.markdown("#### Individual Prediction Explanation")
            
            sample_idx = st.selectbox(
                "Select sample to explain:",
                range(len(X_test)),
                format_func=lambda x: f"Sample {x}: {X_test.index[x]}"
            )
            
            if st.button("Show Explanation"):
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Create explanation object
                if isinstance(shap_values, list):
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_values[1][sample_idx],
                            base_values=st.session_state.shap_explainer.expected_value[1],
                            data=X_test.iloc[sample_idx].values,
                            feature_names=X_test.columns.tolist()
                        ),
                        show=False
                    )
                else:
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_values[sample_idx],
                            base_values=st.session_state.shap_explainer.expected_value,
                            data=X_test.iloc[sample_idx].values,
                            feature_names=X_test.columns.tolist()
                        ),
                        show=False
                    )
                
                st.pyplot(fig)
                plt.close()
    
    with tab4:
        st.subheader("Single Sample Prediction")
        
        if 'ml_model' not in st.session_state:
            st.info("👈 Please train a model first")
            return
        
        st.markdown("""
        Use the trained model to predict the class of new samples based on their 
        microbial profiles. This can be used for clinical risk scoring.
        """)
        
        # Select sample to predict
        all_samples = otu_table.columns.tolist()
        
        sample_to_predict = st.selectbox(
            "Select sample for prediction:",
            all_samples
        )
        
        if st.button("🎯 Predict Sample", type="primary"):
            
            # Get sample data
            sample_data = otu_table[sample_to_predict].to_frame().T
            
            # Ensure same features as training
            train_features = st.session_state.ml_X_train.columns
            sample_data = sample_data[train_features]
            
            # Scale
            sample_scaled = st.session_state.ml_scaler.transform(sample_data)
            sample_scaled = pd.DataFrame(sample_scaled, columns=train_features)
            
            # Predict
            clf = st.session_state.ml_model
            prediction = clf.predict(sample_scaled)[0]
            prediction_proba = clf.predict_proba(sample_scaled)[0]
            
            # Handle label encoding for XGBoost
            if hasattr(clf, 'label_encoder'):
                prediction = clf.label_encoder.inverse_transform([prediction])[0]
                classes = clf.label_encoder.classes_
            else:
                classes = clf.classes_
            
            # Display results
            st.markdown("---")
            st.markdown("#### Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Class", prediction)
            
            with col2:
                max_proba = prediction_proba.max()
                st.metric("Confidence", f"{max_proba:.1%}")
            
            # Probability for each class
            st.markdown("#### Class Probabilities")
            
            proba_df = pd.DataFrame({
                'Class': classes,
                'Probability': prediction_proba
            }).sort_values('Probability', ascending=False)
            
            fig = px.bar(
                proba_df,
                x='Class',
                y='Probability',
                title=f"Prediction Probabilities for {sample_to_predict}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show actual label if available
            if sample_to_predict in metadata.index or sample_to_predict in metadata['SampleID'].values:
                if 'SampleID' in metadata.columns:
                    actual = metadata[metadata['SampleID'] == sample_to_predict][st.session_state.ml_target].values[0]
                else:
                    actual = metadata.loc[sample_to_predict, st.session_state.ml_target]
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Actual class:** {actual}")
                with col2:
                    if actual == prediction:
                        st.success("✅ Correct prediction!")
                    else:
                        st.error("❌ Incorrect prediction")
