#!/usr/bin/env python3
"""Setup script - Run ONCE to initialize everything"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from datetime import datetime, timedelta

# Create directories
Path('data').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)

print("Creating league and team data...")

# League data
leagues_data = {
    'Danish Superligaen': {'historical_draw_rate': 0.28, 'draws_this_season': 47, 'total_matches': 168, 'avg_goals': 2.87, 'strength': 0.92, 'country': 'Denmark'},
    'Turkish Super Lig': {'historical_draw_rate': 0.26, 'draws_this_season': 58, 'total_matches': 224, 'avg_goals': 2.94, 'strength': 0.88, 'country': 'Turkey'},
    'Belgian Pro League': {'historical_draw_rate': 0.24, 'draws_this_season': 45, 'total_matches': 186, 'avg_goals': 2.76, 'strength': 0.85, 'country': 'Belgium'},
    'Swiss Super League': {'historical_draw_rate': 0.25, 'draws_this_season': 42, 'total_matches': 168, 'avg_goals': 2.82, 'strength': 0.87, 'country': 'Switzerland'},
    'Portuguese Primeira Liga': {'historical_draw_rate': 0.23, 'draws_this_season': 43, 'total_matches': 186, 'avg_goals': 2.78, 'strength': 0.86, 'country': 'Portugal'},
    'Netherlands Eredivisie': {'historical_draw_rate': 0.24, 'draws_this_season': 51, 'total_matches': 212, 'avg_goals': 2.91, 'strength': 0.89, 'country': 'Netherlands'},
    'Scottish Premier League': {'historical_draw_rate': 0.22, 'draws_this_season': 39, 'total_matches': 176, 'avg_goals': 2.65, 'strength': 0.83, 'country': 'Scotland'}
}

leagues_df = pd.DataFrame([{'league': name, **info} for name, info in leagues_data.items()])
leagues_df.to_csv('data/leagues_database.csv', index=False)

# Teams data
teams_list = []
for league_name, league_info in leagues_data.items():
    for i in range(14):
        teams_list.append({
            'league': league_name,
            'team': f"{['Ajax', 'Bayern', 'Liverpool', 'Barcelona', 'Real Madrid', 'Juventus', 'PSG', 'City', 'Chelsea', 'United', 'Atletico', 'Arsenal', 'Milan', 'Inter'][i]} {league_name}",
            'draw_rate': np.clip(np.random.normal(league_info['historical_draw_rate'], 0.08), 0.05, 0.40),
            'last_5_draws': np.random.randint(0, 6),
            'home_draw_rate': np.clip(np.random.normal(league_info['historical_draw_rate'], 0.07), 0.05, 0.45),
            'away_draw_rate': np.clip(np.random.normal(league_info['historical_draw_rate'], 0.07), 0.05, 0.35),
            'avg_goals_for': np.random.uniform(1.2, 2.0),
            'avg_goals_against': np.random.uniform(1.2, 2.0),
            'draw_strength': np.random.uniform(0.3, 0.85)
        })

teams_df = pd.DataFrame(teams_list)
teams_df.to_csv('data/teams_database.csv', index=False)

print("Training ML model...")

# Generate training data
np.random.seed(42)
X_train = []
y_train = []

for _ in range(3000):
    league = np.random.choice(list(leagues_data.keys()))
    league_rate = leagues_data[league]['historical_draw_rate']
    
    features = [
        np.random.uniform(0.1, 0.4),  # team_draw_strength
        np.random.uniform(0.1, 0.4),  # opponent_draw_strength
        leagues_data[league]['strength'],  # league_strength
        np.random.randint(0, 6) / 5.0,  # past_5_draws
        np.random.randint(0, 6) / 5.0,  # opponent_past_5_draws
        np.random.uniform(-0.15, 0.15)  # home_advantage
    ]
    
    draw_prob = league_rate + (features[0] + features[1]) / 2 * 0.3 + (features[3] + features[4]) / 2 * 0.2
    draw_prob = np.clip(draw_prob, 0.05, 0.45)
    
    X_train.append(features)
    y_train.append(1 if np.random.random() < draw_prob else 0)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Train model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_scaled, y_train)

joblib.dump(model, 'models/draw_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("✅ All set! Your app is ready to deploy.")
