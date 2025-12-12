"""
Auto-initialize models on app startup
This file is imported by app.py to ensure models exist
"""
import os
import pickle
import numpy as np
from pathlib import Path

def ensure_models_exist():
    """Create models if they don't exist"""
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / 'draw_model.pkl'
    scaler_path = models_dir / 'scaler.pkl'
    
    if not model_path.exists() or not scaler_path.exists():
        create_default_models(models_dir)

def create_default_models(models_dir):
    """Create default models for prediction"""
    
    class DefaultModel:
        """Simple probabilistic model for draw prediction"""
        def predict_proba(self, X):
            n = len(X)
            probs = []
            for features in X:
                # Extract features
                team_strength, opp_strength, league_idx, form1, form2, home_adv = features
                
                # Base probability from league
                base = 0.24 + (league_idx - 0.85) * 0.15
                
                # Team factors
                team_contrib = (team_strength + opp_strength) * 0.15
                
                # Form contribution
                form_contrib = ((form1 + form2) / 2) * 0.1
                
                # Home adjustment
                home_contrib = home_adv * 0.08
                
                # Final probability
                draw_prob = np.clip(base + team_contrib + form_contrib + home_contrib, 0.05, 0.50)
                
                probs.append([1 - draw_prob, draw_prob])
            
            return np.array(probs)
    
    class DefaultScaler:
        """Feature normalization"""
        def transform(self, X):
            means = np.array([0.25, 0.25, 0.87, 0.4, 0.4, 0.0])
            stds = np.array([0.08, 0.08, 0.03, 0.2, 0.2, 0.1])
            return (X - means) / stds
    
    # Create and save
    try:
        model = DefaultModel()
        scaler = DefaultScaler()
        
        with open(models_dir / 'draw_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        with open(models_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        print("✅ Default models created")
    except Exception as e:
        print(f"⚠️ Could not create models: {e}")

# Auto-initialize on import
ensure_models_exist()
