# MAGIC Gamma Telescope Classification with Naive Bayes

A machine learning project implementing probabilistic classification using Naive Bayes to distinguish between gamma rays (signal) and hadron showers (background) using the MAGIC Gamma Telescope dataset.

---

## 📌 Project Overview

This repository demonstrates data loading, exploratory data analysis (EDA), and classification modeling on high-energy astrophysics data from the **MAGIC Gamma Telescope Dataset** (`magic04.data`).

* **Algorithm:** Naive Bayes Classifier
* **Task:** Binary Classification
* **Dataset Size:** 19,020 instances, 11 attributes
* **Class Target:** `g` (gamma signals) vs `h` (hadron background)

---

## 📊 Dataset Description

The dataset consists of 10 continuous numeric features representing image parameters of atmospheric Cherenkov telescope events and 1 target class:

| Feature Name | Description |
| :--- | :--- |
| `fLength` | Major axis of ellipse [mm] |
| `fWidth` | Minor axis of ellipse [mm] |
| `fSize` | 10-log of sum of content of all pixels [in photons] |
| `fConc` | Ratio of sum of two highest pixels over size |
| `fConc1` | Ratio of highest pixel over size |
| `fAsym` | Distance from highest pixel to center, projected onto major axis [mm] |
| `fM3Long` | 3rd root of 3rd moment along major axis [mm] |
| `fM3Trans` | 3rd root of 3rd moment along minor axis [mm] |
| `fAlpha` | Angle of major axis with vector to camera center [deg] |
| `fDist` | Distance from origin to center of ellipse [mm] |
| `class` | Target Class: `g` (gamma) / `h` (hadron) |

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ and the required packages installed:

```bash
pip install pandas numpy scikit-learn
import pandas as pd
import numpy as np

# Define dataset column names
columns = [
    "fLength", "fWidth", "fSize", "fConc", "fConc1",
    "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"
]

# Load dataset
df = pd.read_csv("magic04.data", names=columns)

# Preview data
print(df.head())
print(df.info())# navie-bayes# navie-base
