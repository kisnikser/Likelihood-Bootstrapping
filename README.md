<div align="center">
<h1>Sample Size Determination: Likelihood Bootstrapping </h1>

[Nikita Kiselev](https://github.com/kisnikser)<sup>1 :email:</sup>, [Andrey Grabovoy](https://github.com/andriygav)<sup>1</sup>

<sup>1</sup> Moscow Institute of Physics and Technology

<sup>:email:</sup> Corresponding author

[📝 Paper](https://github.com/kisnikser/Likelihood-Bootstrapping/blob/main/paper/main.pdf), [</> Code](https://github.com/kisnikser/Likelihood-Bootstrapping/tree/main/code)

</div>

## 💡 Abstract
Determining an appropriate sample size is crucial for constructing efficient machine learning models. 
Existing techniques often lack rigorous theoretical justification or are tailored to specific statistical hypotheses about model parameters. 
This paper introduces two novel methods based on likelihood values from resampled subsets to address this challenge. 
We demonstrate the validity of one of these methods in a linear regression model. 
Computational experiments on both synthetic and real-world datasets show that the proposed functions converge as the sample size increases, highlighting the practical utility of our approach.

## 🛠️ Repository Structure
The repository is structured as follows:
- `paper`: This directory contains the main paper in PDF format (`main.pdf`) and the LaTeX source file (`main.tex`). Also there is a directory `figs` with images used in the paper.
- `code`: This directory contains the code used in the paper. It has its own `README.md` file providing a detailed description of the code files.
