# Plant Disease Prediction

An AI-powered plant disease classifier that predicts leaf diseases from images using a CNN and provides optional treatment recommendations through Groq.

## Project layout

```text
plant_disease/
  config.py       Shared paths and model settings
  data/           Dataset validation and image generators
  models/         CNN definition
  prediction/     Model loading and image prediction
  services/       Groq treatment guidance
  training.py     Training and artifact export
app.py            Streamlit interface
PBEL_MODEL_GEN_AI.ipynb  Notebook usage guide
```

## Setup

Use Python 3.11 and install the dependencies from the project root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download and extract the PlantVillage dataset. With the Kaggle CLI configured locally:

```bash
python -m pip install kaggle
kaggle datasets download -d abdallahalidev/plantvillage-dataset --unzip -p data
```

Keep Kaggle credentials outside the repository. The training images should be arranged in class-named subdirectories, for example:

```text
data/
  plantvillage dataset/
    color/
      Apple___healthy/
      Apple___Cedar_apple_rust/
```

The default image directory is `data/plantvillage dataset/color`. Set `PLANT_DISEASE_DATA_DIR` to use another directory.

## Train

From the project root:

```bash
python -m plant_disease.training
```

The script trains for five epochs by default and writes the model and class mapping to `artifacts/`. Override settings as needed:

```bash
python -m plant_disease.training --epochs 10 --data-dir "D:\datasets\plantvillage\color"
```

Set `PLANT_DISEASE_ARTIFACTS_DIR` to change the artifact directory. Keep downloaded datasets and trained models out of Git; both are ignored by `.gitignore`.

## Run the app

For treatment recommendations, create a local `.env` from `.env.example` and set `GROQ_API_KEY`. The key is read from the environment and must never be committed.

```bash
streamlit run app.py
```

The app expects the model and class mapping produced by training in `artifacts/`. Disease treatment guidance requires internet access and a valid Groq API key; leaf classification itself runs locally.

## GitHub

Review `.gitignore` before staging files. Do not commit credentials, `kaggle.json`, datasets, virtual environments, or generated model artifacts.
