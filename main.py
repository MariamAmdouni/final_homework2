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
    runs-on: ubuntu-latest
    
    permissions:
      contents: write
      pull-requests: write
      repository-projects: write
      id-token: write 

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v3
        
      - name: Set Up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install mlflow torch torchvision

      - name: Set up DVC and Credentials
        env:
          GDRIVE_CREDENTIALS_DATA: ${{ secrets.GDRIVE_CREDENTIALS_DATA }}
        run: |
          echo $GDRIVE_CREDENTIALS_DATA > credentials.json
          dvc remote modify --local gdrive-remote gdrive_service_account_json_file_path credentials.json
          dvc pull -v

      - name: Train CNN Model (Initial 30 Epochs)
        run: |
          python main.py --mode train --data_path ./data/train/

      - name: Fine-Tune - Experiment 1
        run: |
          python main.py --mode train --data_path ./data/train/ --epochs 20 --batch_size 8

      - name: Fine-Tune - Experiment 2
        run: |
          python main.py --mode train --data_path ./data/train/ --epochs 20 --batch_size 16

      - name: Fine-Tune - Experiment 3
        run: |
          python main.py --mode train --data_path ./data/train/ --epochs 25 --batch_size 16

      - name: Test and Log with MLflow
        run: |
          python main.py --mode test --data_path ./data/test/ --model_path ./models/cnn_model.pth

