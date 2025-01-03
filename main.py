name: Train and Fine-Tune CNN

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  train-cnn:
    runs-on:ubuntu-latest
    
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v3
        
      - name: Set Up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'  # Use valid Python version

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
