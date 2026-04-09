name: Parallel Matrix Training (CPU)

on:
  workflow_dispatch:

env:
  HF_TOKEN: ${{ secrets.HF_TOKEN }}
  INPUT_DATASET: "P2SAMAPA/fi-etf-macro-signal-master-data"
  OUTPUT_DATASET: "P2SAMAPA/p2-etf-sdf-engine-results"

jobs:
  train:
    runs-on: ubuntu-latest
    strategy:
      max-parallel: 4
      matrix:
        fold: [0, 1, 2, 3]
        lr: [0.001, 0.01]
        model: ["rf", "xgb"]

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Cache pip
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
      - name: Install dependencies
        run: |
          pip install datasets huggingface_hub pandas numpy pyyaml scikit-learn xgboost
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
      - name: Login to HF
        run: huggingface-cli login --token ${{ env.HF_TOKEN }}
      - name: Run training
        run: |
          python train.py \
            --input ${{ env.INPUT_DATASET }} \
            --output ${{ env.OUTPUT_DATASET }} \
            --fold ${{ matrix.fold }} \
            --lr ${{ matrix.lr }} \
            --model ${{ matrix.model }}
