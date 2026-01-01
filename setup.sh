#!/bin/bash
# Setup script for NYC Tax RAG System
# Requires Python 3.10+

set -e

echo "=== NYC Tax RAG System Setup ==="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "${PYTHON_VERSION%.*}" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python 3.10+ is required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "Python version: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env from example if it doesn't exist
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "Creating .env from .env.example..."
        cp .env.example .env
        echo "Please update .env with your API keys."
    fi
fi

echo ""
echo "=== Setup Complete ==="
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Don't forget to:"
echo "  1. Update your .env file with API keys"
echo "  2. Place your tax_law.txt in the data/ folder"
