import os
import sys
from pathlib import Path

# Create directories if they don't exist
Path('data').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)

print("🔧 Initializing Draw Predictor Pro...")

# Step 1: Build databases
print("\n📊 Creating league and team databases...")
from data_builder import create_league_database
leagues_df, teams_df, matches_df = create_league_database()
print("✅ Databases created!")

# Step 2: Train model
print("\n🤖 Training prediction model...")
from model_builder import train_model
try:
    model, scaler = train_model()
    print("✅ Model trained and saved!")
except Exception as e:
    print(f"⚠️ Model training warning: {e}")
    print("   App will still work with rule-based logic")

print("\n" + "="*50)
print("✨ INITIALIZATION COMPLETE!")
print("="*50)
print("\n🚀 Ready to deploy on Streamlit Cloud!")
print("\nNext steps:")
print("1. Push this folder to GitHub")
print("2. Go to https://streamlit.io/cloud")
print("3. Deploy the app.py file")
print("4. Share your live prediction link!")
