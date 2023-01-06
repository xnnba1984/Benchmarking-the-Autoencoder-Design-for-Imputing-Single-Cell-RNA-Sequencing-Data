# Benchmarking-the-Autoencoder-Design-for-Imputing-Single-Cell-RNA-Sequencing-Data

This repository contains the R and Python code to create the result in the paper 'Benchmarking the Autoencoder Design for Imputing Single-Cell RNA Sequencing Data'. The data used in this study is available at [Zenodo repository](https://zenodo.org/record/7504311#.Y7ei0naZMuK).

## Code Structure

### 1. data process random mask.R

The R code to conduct random masking on 12 real scRNA-seq datasets.

### 2. data process double median mask.R

The R code to conduct median masking on 12 real scRNA-seq datasets.

### 3. data process double exponential mask.R

The R code to conduct double exponential masking on 12 real scRNA-seq datasets.

### 4. data process clustering.R

The R code to conduct k-means clustering and compute clustering accuracy on 20 real scRNA-seq datasets before imputation. The same code can also be applied to datasets after imputation.

### 5. data process DE.R

The R code to simulate data with ground-truth DE genes from real datasets.It conducts DE gene analysis and computes performance before and after imputation.

### 6. overall imputation accuracy compare.R

The R code to compute RNMSE and correlation of masking values between original and imputed data.

### 7. depth and width.py

The Python code to train autoencoders with different depths and widths. The trained model will impute the dataset. The training and imputation are implemented by Pytorch deep learning library.

### 8. activation function.py

The Python code to train autoencoders with different activation functions. The trained model will impute the dataset. The training and imputation are implemented by Pytorch deep learning library.

### 9. regularization.py

The Python code to train autoencoders with different levels of weight decay and droput regularization. The trained model will impute the dataset. The training and imputation are implemented by Pytorch deep learning library.
