# Tutorial: Introduction to Hyperbolic Random Forests (HoroRF)

## Disclaimer

This tutorial is based on [Hyperbolic Random Forests](https://arxiv.org/abs/2308.13279) research paper.

## 1. Introduction

**What is the problem?**

Many real-world datasets, such as biological taxonomies, social networks, and language hierarchies, exhibit a *hierarchical structure*. When we embed this data into a standard Euclidean space (like a 2D plane or 3D cube), we often distort the relationships between points, especially as the hierarchy grows deep.

**What is the solution?**

*Hyperbolic space* is a non-Euclidean geometry that naturally accommodates exponential growth, making it perfect for representing trees and hierarchies. However, standard machine learning algorithms (like SVMs or Random Forests) are designed for Euclidean lines and planes.

**What is HoroRF?**

**HoroRF** (Hyperbolic Random Forest) is an adaptation of the classic Random Forest algorithm to operate directly in hyperbolic space. Instead of splitting data using straight lines (hyperplanes), it splits data using **horospheres**—the hyperbolic equivalent of planar boundaries.

For a deeper visual explanation of Random Forests in general, which helps in understanding the ensemble nature of this algorithm, I recommend the following video:

[StatQuest: Random Forests Part 1](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)

This video is relevant because it explains the foundational concepts of Random Forests (bootstrapping, feature randomness, and voting) which HoroRF extends into hyperbolic geometry.

## 2. Core Concepts

To understand HoroRF, we must define a few key terms without getting bogged down in complex differential geometry.

### 2.1 The Poincaré Ball Model

Think of the hyperbolic space as a unit disk (the Poincaré ball).

* **Center**: The origin represents the "root" of a hierarchy.
* **Edge**: The boundary of the disk represents infinity.
* **Distance**: As you move toward the edge, distances grow exponentially. This allows the space to "fit" massive trees that would run out of room in a Euclidean circle.

### 2.2 The Horosphere (The Splitter)

In a standard Decision Tree, we split data using a line: .
In HoroRF, we split data using a **Horosphere**.

* Visually, in the Poincaré disk, a horosphere looks like a Euclidean circle that is tangent to the boundary of the disk.
* Mathematically, it is defined by a normal vector  (an ideal point on the boundary) and a scalar offset .
* **Decision Rule**: A point  is sent to the left or right child node based on whether it is "inside" or "outside" this horosphere.

### 2.3 HoroSVM

Finding the *perfect* horosphere to split data is computationally hard. HoroRF uses a "large-margin classifier" called **HoroSVM** at each node to find a good candidate split quickly. It tries to separate the classes with the widest possible margin using a horosphere.

## 3. The Algorithm

The training process of HoroRF mirrors the classic Random Forest but replaces Euclidean operations with hyperbolic ones:

1. **Bootstrap**: Select a random subset of the training data.
2. **Tree Construction**:
  - At each node, consider a subset of features (dimensions).
  - **Find Split**: Use **HoroSVM** to find a horosphere that separates the classes.
  - **Optimization**: If the data is multi-class, HoroRF groups classes based on their **Lowest Common Ancestor (LCA)** in the hierarchy to turn it into a binary problem for the split.
  - **Partition**: Move points to left/right child nodes based on the horosphere decision rule.
  - Repeat until a max depth is reached or the leaf is pure.
3. **Ensemble**: Train multiple trees ().
4. **Prediction**: Aggregate votes from all trees to classify new points.

## 4. Practical Implementation

We will use the official implementation provided by the authors.

### 4.1 Prerequisites

You need a Python environment with PyTorch and standard scientific computing libraries. All packages are listed in `Pipfile` which can be install using `Pipenv` application.

- Clone the repository `git clone https://github.com/LarsDoorenbos/HoroRF.git`
- Change directory to the package `cd HoroRF`
- Install requirements `pipenv install`

### 4.2 Data Format

HoroRF expects data already embedded in hyperbolic space (Poincaré ball).

* **Features**: A matrix of shape `(N, D)` where values are within the unit ball ().
* **Labels**: A vector of shape `(N,)` with class integers.

### 4.3 Running the Code

The repository uses a `params.yml` file to control hyperparameters, which is excellent for reproducibility.

**Step 1: Configure `params.yml**`
Create or edit the `params.yml` file in the root directory:

```yaml
dataset_file: datasets.karate  # Example dataset included in repo
seed: 42
num_jobs: 1                    # Number of parallel jobs
# HoroRF parameters
criterion: 'gini'              # Split quality measure
max_depth: 6                   # Depth of trees
num_trees: 10                  # Number of trees in forest
visualize: no                  # Set to 'yes' to save plots

```

**Step 2: Training Script**
The main entry point is `train_hyp_rf.py`. You can run it directly:

```bash
python train_hyp_rf.py

```

**Step 3: Custom Python Usage (Conceptual)**
If you want to integrate it into your own script, the structure generally follows the Scikit-Learn API. Below is an example of how you would instantiate and train the model programmatically (based on typical usage of the codebase):

```python
import torch
import numpy as np
# Import the HoroRF class (verify path in repo structure, likely hororf/hororf.py)
from hororf.hororf import HoroRF 

# 1. Generate synthetic hyperbolic data (toy example)
# In practice, load your pre-trained Poincare embeddings
X_train = np.random.uniform(-0.5, 0.5, (100, 2))  # Ensure norm < 1
y_train = np.random.randint(0, 2, 100)

X_test = np.random.uniform(-0.5, 0.5, (20, 2))

# 2. Initialize Model
# Note: 'C' is the regularization parameter for the internal HoroSVM
model = HoroRF(
    num_trees=10,
    max_depth=5,
    criterion='gini',
    C=1.0 
)

# 3. Fit and Predict
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Predictions: {predictions}")

```

## 5. Summary of Contribution

* **Why use it?** If your data is hierarchical (e.g., word embeddings, biological trees) and you are using hyperbolic embeddings, HoroRF will likely outperform Euclidean Random Forests by respecting the geometry of the data.
* **Key Insight**: The "straight lines" of Euclidean space are replaced by "horospheres" in Hyperbolic space to create decision boundaries.

