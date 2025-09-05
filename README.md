# sign2text-api

API microservice for sign2text backend.

# Prerequisites

1. Python
2. pip

# Running the app

Install NVCC and pytorch:

Create conda environment:

```bash
conda create --name sign2text-api python=3.11
conda activate sign2text-api
# conda install pytorch==2.8.0 torchvision==0.12.6 torchaudio==2.6.0 cudatoolkit=12.6 -c pytorch -c conda-forge
```

```bash
conda install nvidia::cuda-toolkit
nvcc --version # ensure torch and nvcc version match
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
```

```
# Start the server (within the environment)
fastapi dev main.py
```

# Documentation

```
# After server started
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc
```
