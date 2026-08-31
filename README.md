# Efficient Non-Local Means Image Denoising

Implementation and experimental evaluation of an efficient Non-Local Means (NLM) image denoising algorithm.

## Objective

Implement a computationally efficient version of the basic Non-Local Means algorithm while maintaining comparable denoising quality.

## Methods

1. Basic Non-Local Means
2. Efficient Non-Local Means using integral images

## Evaluation

The methods are compared using:

- PSNR
- SSIM
- Runtime
- Speedup

Four images from the USC SIPI Miscellaneous Image Database are used for evaluation.

## Project Structure

```text
efficient-nlm/
├── src/
├── data/
├── results/
├── figures/
├── report/
├── requirements.txt
└── README.md

