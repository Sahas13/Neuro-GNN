# Dynamic GNN Seizure Detection

This repository contains a minimal project that trains a small graph neural network (GNN) on EEG samples and exposes a simple Flask API to predict seizure vs normal EEG. It also includes a lightweight frontend dashboard to visualize EEG channels.

Contents
- `backend/` — Flask app, model, and training script
- `dataset/` — example EEG CSV (`eeg_sample.csv`)
- `frontend/` — static dashboard (`index.html`)
- `.gitignore` — recommended ignores

Minimum requirements
- Windows (PowerShell)
- Python 3.8+ recommended (3.10/3.11 tested)
- pip
- (Optional) GPU + CUDA for faster training — adjust PyTorch install accordingly

Quick start (PowerShell)

1) Create and activate a virtual environment

```powershell
cd C:\Users\sahas\OneDrive\Desktop\Sahas_SIST
python -m venv .venv
# Activate
.\.venv\Scripts\Activate.ps1
```

2) Install Python dependencies

The project uses PyTorch and PyTorch Geometric. Installing PyTorch Geometric is platform and CUDA-version specific. The simplest CPU-only example is shown below. If you have CUDA, follow the installation guide at https://pytorch.org and https://pytorch-geometric.readthedocs.io.

CPU (example):

```powershell
# Install PyTorch (CPU build) and other packages
pip install --upgrade pip
pip install "torch==2.1.0+cpu" torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt

# Install PyTorch Geometric dependencies (example CPU wheels)
# Note: Replace versions with those matching your torch install if needed
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-geometric
```

If you prefer, follow the official PyG installation instructions to pick the correct wheel URLs:
- https://pytorch.org
- https://pytorch-geometric.readthedocs.io

3) Train the model (optional)

This project includes a tiny sample dataset in `dataset/eeg_sample.csv`. To train a model and produce `seizure_gnn.pth`:

```powershell
# Run from repository root or from the backend folder
python backend/train_model.py
```

This will save `seizure_gnn.pth` in the `backend/` working directory when finished. Training in this example is minimal and meant for demonstration only.

4) Run the backend Flask API

Make sure the model file `seizure_gnn.pth` is present in the `backend/` folder. Then run:

```powershell
python backend/app.py
```

The API will start on `http://127.0.0.1:5000` by default.

5) Test the prediction endpoint

You can POST a JSON payload with an `eeg` array of channel values. Example using PowerShell (curl is available as `curl` or `Invoke-RestMethod`):

```powershell
# Using curl
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"eeg": [12.3,11.5,10.8,9.9,8.1,7.6,6.2,5.8]}'

# Using PowerShell Invoke-RestMethod
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method POST -Body (@{ eeg = @(12.3,11.5,10.8,9.9,8.1,7.6,6.2,5.8) } | ConvertTo-Json) -ContentType 'application/json'
```

The response JSON will include `prediction` (human text) and `class` (0 or 1).

6) Run the frontend UI

Open the static frontend in a browser. For best results, serve it over HTTP instead of opening the file directly.

```powershell
cd frontend
python -m http.server 8000
# Then open http://localhost:8000 in your browser
```

The UI will send data manually entered by you to the backend if you wire the frontend's JS to call the prediction endpoint. Currently the provided `index.html` runs a local analytic routine; you can extend it to call the Flask API using fetch.

Example fetch request (to add in frontend JS):

```javascript
fetch('http://127.0.0.1:5000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ eeg: [12,11,10,9,8,7,6,5] })
})
.then(r => r.json()).then(console.log).catch(console.error)
```

Notes and troubleshooting
- If you see errors importing `torch_geometric` or its dependencies, ensure you installed the correct wheel builds for your installed `torch` version and your CUDA/CPU configuration.
- If the Flask app cannot find `seizure_gnn.pth`, ensure you are running `python backend/app.py` from the repository root or that the model file exists in the current working directory.
- The included dataset is tiny and only for demonstration — results will not be clinically meaningful.

License
- This repository contains example code and is provided as-is for learning and prototyping.

If you want, I can:
- Add a small script to the frontend to call the backend prediction endpoint
- Add a Makefile / PowerShell script to automate env setup and launches
- Create GitHub Actions to run linting or run tests

