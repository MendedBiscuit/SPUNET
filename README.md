<div align="center">
  <img src="img/logo.png" alt="SPUNET Logo" width="600">

  **UNET model and training data used to used to detect impurities in samples of wood chips.**

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  ![Version](https://img.shields.io/badge/version-1.0.0-blue)
</div>

---

## Overview
SPUNET ...

## Installation

```bash
# Clone the repository
git clone https://github.com/MendedBiscuit/spunet.git

# Navigate to the directory
cd spunet

# Initiate venv and install dependencies using uv
uv venv
uv pip install -r requirements.txt

## Usage

# Training
To train the UNET, run train.py for the desired duration

# Predicting
Predicting the location of impurities can be done by specifing the location of the test images in predict.py and subsequently running the file.
