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
- Install the required packages from `method_name_prediction\requirements.txt`
- Download a few Java repositories and place them in `data\repos\training_repos`, `data\repos\testing_repos`, and `data\repos\val_repos` for training, testing, and validation respectively.
- Execute `preprocess.sh`. The script first creates AST of each Java file, extracts AST path, and removes paths excepts full contexts and partial contexts.
- Execute `train.sh`

## ML Essentials
- Create a virtual environment with Python>=3.9
- Install the required packages from `ml-essentials\requirements.txt`
- Run notebooks in the subfolders of `ml-essentials`; all notebooks are independent from each other.