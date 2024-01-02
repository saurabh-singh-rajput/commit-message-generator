# Automated Commit Message Generation

## Overview
In this repository, we present a project designed to generate automated commit messages using deep learning, focusing on the practical implementation of an automated commit message generation system using fine-tuning techniques on a pre-trained language model.

## Project Structure

- `.gitignore` - Specifies untracked files that Git should ignore.
- `README.md` - This file, containing documentation for the repository.
- `commit_message_generator.ipynb` - Jupyter notebook with the implementation of the commit message generator.
- `filtered_data.csv` - A dataset file containing pre-processed commit data for training, that will be used for training and evaluation.
- `phi2_finetune_own_data.ipynb` - Jupyter notebook for fine-tuning Microsoft's Phi-2 model on commit message generation, that you can easily use for your own dataset.
- `requirements.txt` - A list of Python dependencies required by the project.
- `test.3000.diff` - Sample diff file containing code changes for testing.
- `test.3000.msg` - Sample commit message file corresponding to the `test.3000.diff` file.

## Prerequisites

Before setting up the project, ensure you have the following:

- Python 3.8 or higher
- pip (Python package installer)
- Access to Google Colab Free Tier (for running Jupyter notebooks with GPU support)

## Setup

1. **Clone the Repository:**

   Clone the repository to your local machine or a server where you plan to run the project:

   ````shell
   git clone https://github.com/saurabh-singh-rajput/commit-message-generator.git
   cd commit-message-generator
   ```

2. **Install Dependencies:**

   Install the necessary Python packages using `pip`:

   ````shell
   pip install -r requirements.txt
   ```
  Note : Google colab may throw issues when installing the dependencies of this `requirements.txt` dues to numpy version conflict. In that scenario, Just simply disconnect and restart the collab runtime, and simply run the notebook without the `pip install -r requirements.txt`, as i have added the required dependencies installation in the code, which should take care of it, and run successfully.

## Running the Notebooks

1. **commit_message_generator.ipynb:**

   This notebook contains the core logic for generating commit messages. To run it:

   - Open Google Colab: https://colab.research.google.com/
   - Upload the `commit_message_generator.ipynb` file to Colab.
   - Follow the instructions within the notebook to run the cells, which include data preparation, model setup, training, and message generation steps.

2. **phi2_finetune_own_data.ipynb:**

   This notebook guides you through fine-tuning Microsoft's Phi-2 model on your dataset. I have used the same commit message dataset for generation in this scenario. To use it:

   - Open Google Colab: https://colab.research.google.com/
   - Upload the `phi2_finetune_own_data.ipynb` file to Colab.
   - Execute the instructions step-by-step, which covers the environment setup, data formatting, and the fine-tuning process.

## Usage

1. **Data Preparation:**

   Prepare your dataset according to the format explained in the `phi2_finetune_own_data.ipynb` and `commit_message_generator.ipynb` notebooks. Once the data is read from the CSV, we convert it into Pyarrow datatype as shown in the code.

2. **Training the Model:**

   Follow the steps in `phi2_finetune_own_data.ipynb` and `commit_message_generator.ipynb` to train the model on your prepared dataset. Ensure to adjust the hyperparameters according to your dataset size and complexity.

3. **Generating Commit Messages:**

   Once the model is trained, use the inference steps in the notebook to generate commit messages for new code diffs.

## Contribution

We encourage contributions to this project. If you want to improve the notebooks or add more features, please fork the repository, make your changes, and create a pull request.

<!---
## License

This project is open-sourced under the MIT license.
-->

## Acknowledgements

We would like to thank the authors of the original notebooks and datasets that have made this educational project possible.
