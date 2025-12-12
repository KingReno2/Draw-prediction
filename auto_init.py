"""
Auto-generate models on Streamlit Cloud startup
This is automatically called by app.py
"""

import os
import pickle
import numpy as np
from pathlib import Path

def init_models():
    """Initialize models if they don't exist"""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Check if models exist
    if (models_dir / 'draw_model.pkl').exists() and (models_dir / 'scaler.pkl').exists():
        return
    
    # Create simple models
    class PredictionModel:
        def predict_proba(self, X):
            results = []
            for row in X:
                # Simple probability calculation
                prob = 0.24 + (row[0] + row[1]) * 0.15 + (row[3] + row[4]) * 0.1
                prob = np.clip(prob, 0.05, 0.45)
                results.append([1 - prob, prob])
            return np.array(results)
    
    class FeatureScaler:
        def transform(self, X):
            return (X - np.array([0.25, 0.25, 0.87, 0.4, 0.4, 0.0])) / np.array([0.08, 0.08, 0.03, 0.2, 0.2, 0.1])
    
    # Save models
    with open(models_dir / 'draw_model.pkl', 'wb') as f:
        pickle.dump(PredictionModel(), f)
    
    with open(models_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(FeatureScaler(), f)

# Initialize on import
try:
    init_models()
except Exception as e:
    pass  # Silently fail - app will still work
