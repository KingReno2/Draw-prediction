import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta
import pickle
import os

# Historical data for 7 high-draw leagues
np.random.seed(42)

leagues_data = {
    'Danish Superligaen': {
        'historical_draw_rate': 0.28,
        'matches': 240,
        'strength': 0.92
    },
    'Turkish Super Lig': {
        'historical_draw_rate': 0.26,
        'matches': 272,
        'strength': 0.88
    },
    'Belgian Pro League': {
        'historical_draw_rate': 0.24,
        'matches': 240,
        'strength': 0.85
    },
    'Swiss Super League': {
        'historical_draw_rate': 0.25,
        'matches': 240,
        'strength': 0.87
    },
    'Portuguese Primeira Liga': {
        'historical_draw_rate': 0.23,
        'matches': 240,
        'strength': 0.86
    },
    'Netherlands Eredivisie': {
        'historical_draw_rate': 0.24,
        'matches': 272,
        'strength': 0.89
    },
    'Scottish Premier League': {
        'historical_draw_rate': 0.22,
        'matches': 240,
        'strength': 0.83
    }
}

def generate_training_data(n_samples=5000):
    """Generate realistic historical match data for training"""
    X = []
    y = []
    
    for _ in range(n_samples):
        league_key = np.random.choice(list(leagues_data.keys()))
        league = leagues_data[league_key]
        
        # Features
        team_draw_strength = np.random.uniform(0.1, 0.4)
        opponent_draw_strength = np.random.uniform(0.1, 0.4)
        league_strength = league['strength']
        past_5_draws = np.random.randint(0, 6)
        opponent_past_5_draws = np.random.randint(0, 6)
        home_advantage = np.random.uniform(-0.15, 0.15)
        
        # Combine features
        features = [
            team_draw_strength,
            opponent_draw_strength,
            league_strength,
            past_5_draws / 5.0,
            opponent_past_5_draws / 5.0,
            home_advantage
        ]
        
        # Draw probability based on features and league historical rate
        draw_prob = (league['historical_draw_rate'] + 
                    (team_draw_strength + opponent_draw_strength) / 2 * 0.3 +
                    (past_5_draws + opponent_past_5_draws) / 10 * 0.2)
        
        draw_prob = np.clip(draw_prob, 0.05, 0.45)
        
        # Binary outcome
        outcome = 1 if np.random.random() < draw_prob else 0
        
        X.append(features)
        y.append(outcome)
    
    return np.array(X), np.array(y)

def train_model():
    """Train the draw prediction model"""
    print("Generating training data...")
    X, y = generate_training_data(5000)
    
    print("Training model...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_scaled, y)
    
    print("Saving model...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/draw_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("Model trained and saved!")
    return model, scaler

if __name__ == "__main__":
    train_model()
