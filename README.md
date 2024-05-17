# Sample Size Determination: Likelihood Bootstrapping

**Author:** Nikita Kiselev

**Advisor:** Andrey Grabovoy

## Abstract
The problem of determining an appropriate sample size is essential for constructing an efficient machine learning model. 
However, current techniques are either not rigorously proven or are specific to a particular statistical hypothesis regarding the distribution of model parameters. 
In this paper we propose two methods based on the likelihood values on resampled subsets. 
We demonstrate the validity of one of these methods in a linear regression model. 
Computational experiments show the convergence of the proposed functions as the sample size increases.

## Repository Structure
The repository is structured as follows:
- `paper`: This directory contains the main paper in PDF format (`main.pdf`) and the LaTeX source file (`main.tex`). Also there is a directory `figs` with images used in the paper.
- `code`: This directory contains the code used in the paper. It has its own `README.md` file providing a detailed description of the code files.
