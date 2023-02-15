# ML4SE course

## Case study 1: Code smell detection
Two variants
- smell detection using classic ML techniques
- smell detection using DL (specifically Autoencoder)

Both variants share some steps/scripts.

### Smell detection using classic ML
- `repo_download.ipynb`
- `analyze_code.ipynb`
- `create_dataset_ML.ipynb`
- `train_ML.ipynb`

### Smell detection using DL (Autoencoder)
- `repo_download.ipynb`
- `analyze_code.ipynb`
- `analyze_code_codesplit.ipynb`
- `create_dataset_DL.ipynb`
- `tokenize.ipynb`
- `train_autoencoder.ipynb`

## Case study 2: Method name prediction
The folder `method_name_prediction` contains the *Code2Vec* implementation.
The original repository can be found on [GitHub](https://github.com/tech-srl/code2vec)

- Create a virtual environment with Python>=3.9
- Install the required packages from `requirements.txt`