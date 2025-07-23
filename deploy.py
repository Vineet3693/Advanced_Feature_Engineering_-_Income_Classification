
import subprocess
import sys
import os

def setup_and_run():
    """Setup and run the Streamlit app"""
    
    print("Setting up Income Classification Model...")
    
    # Install requirements
    print("Installing requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Train model if not exists
    if not os.path.exists('income_model.pkl'):
        print("Training model...")
        exec(open('train_model.py').read())
    
    # Run streamlit app
    print("Starting Streamlit app...")
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    setup_and_run()
