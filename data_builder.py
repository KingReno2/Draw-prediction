import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle

def create_league_database():
    """Create comprehensive league statistics database"""
    
    leagues = {
        'Danish Superligaen': {
            'country': 'Denmark',
            'historical_draw_rate': 0.28,
            'draws_this_season': 47,
            'total_matches': 168,
            'avg_goals': 2.87,
            'strength': 0.92
        },
        'Turkish Super Lig': {
            'country': 'Turkey',
            'historical_draw_rate': 0.26,
            'draws_this_season': 58,
            'total_matches': 224,
            'avg_goals': 2.94,
            'strength': 0.88
        },
        'Belgian Pro League': {
            'country': 'Belgium',
            'historical_draw_rate': 0.24,
            'draws_this_season': 45,
            'total_matches': 186,
            'avg_goals': 2.76,
            'strength': 0.85
        },
        'Swiss Super League': {
            'country': 'Switzerland',
            'historical_draw_rate': 0.25,
            'draws_this_season': 42,
            'total_matches': 168,
            'avg_goals': 2.82,
            'strength': 0.87
        },
        'Portuguese Primeira Liga': {
            'country': 'Portugal',
            'historical_draw_rate': 0.23,
            'draws_this_season': 43,
            'total_matches': 186,
            'avg_goals': 2.78,
            'strength': 0.86
        },
        'Netherlands Eredivisie': {
            'country': 'Netherlands',
            'historical_draw_rate': 0.24,
            'draws_this_season': 51,
            'total_matches': 212,
            'avg_goals': 2.91,
            'strength': 0.89
        },
        'Scottish Premier League': {
            'country': 'Scotland',
            'historical_draw_rate': 0.22,
            'draws_this_season': 39,
            'total_matches': 176,
            'avg_goals': 2.65,
            'strength': 0.83
        }
    }
    
    # Create team statistics
    teams_data = []
    
    for league_name, league_info in leagues.items():
        # Generate 12-16 teams per league
        num_teams = np.random.randint(12, 17)
        
        for i in range(num_teams):
            team_name = f"Team_{i+1}_{league_name.replace(' ', '_')}"
            
            teams_data.append({
                'league': league_name,
                'team': team_name,
                'draw_rate': np.clip(np.random.normal(league_info['historical_draw_rate'], 0.08), 0.05, 0.40),
                'last_5_draws': np.random.randint(0, 6),
                'home_draw_rate': np.clip(np.random.normal(league_info['historical_draw_rate'], 0.07), 0.05, 0.45),
                'away_draw_rate': np.clip(np.random.normal(league_info['historical_draw_rate'], 0.07), 0.05, 0.35),
                'avg_goals_for': np.random.uniform(1.2, 2.0),
                'avg_goals_against': np.random.uniform(1.2, 2.0),
                'draw_strength': np.random.uniform(0.3, 0.85)
            })
    
    teams_df = pd.DataFrame(teams_data)
    teams_df.to_csv('data/teams_database.csv', index=False)
    
    # Create leagues summary
    leagues_df = pd.DataFrame([
        {'league': name, **info}
        for name, info in leagues.items()
    ])
    leagues_df.to_csv('data/leagues_database.csv', index=False)
    
    # Create sample recent matches
    matches = []
    today = datetime.now()
    
    for league_name in leagues.keys():
        league_teams = teams_df[teams_df['league'] == league_name]['team'].tolist()
        
        for _ in range(8):
            home_team = np.random.choice(league_teams)
            away_team = np.random.choice(league_teams)
            
            if home_team == away_team:
                continue
            
            matches.append({
                'league': league_name,
                'date': today - timedelta(days=np.random.randint(1, 90)),
                'home_team': home_team,
                'away_team': away_team,
                'result': np.random.choice(['W', 'D', 'L'], p=[0.4, 0.25, 0.35]),
                'home_goals': np.random.randint(0, 4),
                'away_goals': np.random.randint(0, 4)
            })
    
    matches_df = pd.DataFrame(matches)
    matches_df.to_csv('data/historical_matches.csv', index=False)
    
    print("✅ League database created successfully!")
    return leagues_df, teams_df, matches_df

if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    create_league_database()
