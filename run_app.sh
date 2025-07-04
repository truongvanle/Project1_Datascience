#!/bin/bash
# Simple script to run the Streamlit application with virtual environment

echo "🚀 Starting ITViec Analytics Platform..."
echo "Activating virtual environment..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "📦 Installing requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "✅ Virtual environment found. Activating..."
    source .venv/bin/activate
fi

echo "🎯 Starting Streamlit application..."
echo "Access the app at: http://localhost:8503"
echo ""

# Check for common missing dependencies and install if needed
echo "🔍 Checking dependencies..."
source .venv/bin/activate
pip install -r requirements.txt --quiet

# Run the Streamlit app
streamlit run app.py --server.port 8503 --server.address 0.0.0.0
