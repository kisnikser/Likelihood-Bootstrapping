# Guide

This directory contains the source code for the paper "Sample Size Determination: Likelihood Bootstrapping". The code includes data processing, model building, and visualization of results.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Files Description](#files-description)

## Installation <a name="installation"></a>
To use this project, you need to have Python and Jupyter Notebook installed on your computer. You can download Python from [here](https://www.python.org/downloads/) and Jupyter Notebook from [here](https://jupyter.org/install).

Clone the repository:
```bash
git clone https://github.com/kisnikser/Likelihood-Bootstrapping.git
```

Navigate to the code directory:
```bash
cd code
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage <a name="usage"></a>
To run the project, open the Jupyter Notebook:
```bash
jupyter notebook
```
Then, open the `main.ipynb` file in the Jupyter Notebook interface.

## Files Description <a name="files-description"></a>
- `data.py`: contains functions for loading and preprocessing data.
- `utils.py`: contains utility functions, including for loading and saving binary files.
- `stuff.py`: contains various functions used in the project.
- `models.py`: contains classes of the probability models used.
- `visualize.py`: contains functions for visualizing data and results.
- `main.ipynb`: the main Jupyter Notebook file, containing the project's code.
