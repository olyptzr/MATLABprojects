# Machine Learning & Computational Physics (MATLAB)

This repository features 6 comprehensive projects focusing on the implementation of advanced computational models for physical systems and the application of machine learning for engineering data analysis.

---

## Projects Overview

### 1. Advanced Data Preprocessing Toolkit
* **Core Task:** Developed a robust pipeline for cleaning and normalizing sensor and physics data.
* **Implementation:** Handled missing values via linear/spline interpolation, detected outliers using the **Interquartile Range (IQR)** method, and implemented Min-Max, Z-score, and Robust scaling.
* **Visualization:** Generated correlation and covariance heatmaps to analyze feature relationships and joint variability.

### 2. K-Nearest Neighbors (kNN) & Performance Metrics
* **Core Task:** Classification of orbital mechanics (bound vs. unbound orbits) and radar data (Ionosphere dataset).
* **Implementation:** Developed a custom kNN regression function and a performance evaluation suite.
* **Optimization:** Used **Bayesian Hyperparameter Optimization** and Grid Search with K-fold cross-validation to maximize Accuracy and F1-score while minimizing MAPE for regression tasks.

### 3. Decision Trees & Support Vector Machines (SVM)
* **Core Task:** Classification of astronomical data from the **MAGIC Gamma Ray Telescope**.
* **Implementation:** Built and compared Decision Tree and SVM models using **Linear and RBF kernels**.
* **Regression:** Applied **Support Vector Regression (SVR)** and Regression Trees for engineering signal path-loss prediction, utilizing RMSE and MAPE for model validation.

### 4. Neural Networks for Non-Linear Physics
* **Core Task:** Predicting the oscillation period of a pendulum at large angles where standard linear approximations ($T \approx 2\pi\sqrt{L/g}$) fail.
* **Architecture:** Designed a Feedforward Artificial Neural Network (ANN) using **Levenberg-Marquardt backpropagation**.
* **Preprocessing:** Utilized Z-score standardization and early stopping to ensure high model generalization and prevent overfitting.

### 5. Thermal Management & Heat Transfer Library
* **Core Task:** Created a library of functions to calculate heat flow through **Conduction, Convection, and Radiation**.
* **Simulation:** Modeled multi-layer composite wall systems (Concrete, Insulation, Gypsum) to determine total thermal resistance and precise interface temperatures.
* **Steady-State Analysis:** Validated power loss across various environmental conditions.

### 6. Stochastic Simulations & Algorithmic Complexity
* **Core Task:** Monte Carlo-style simulation of nanoparticle movement in partitioned systems.
* **Physics:** Modeled random walk transitions and reached system equilibrium using stochastic probability rules.
* **Computational Analysis:** Conducted CPU time scaling analysis to determine computational complexity ($O(N^p)$) using log-log fitting and verified statistical convergence with error bars over multiple runs.

---

## Repository Contents

### Main Scripts (Run these)
* **`exercise_1_preprocessing.m`**: Data cleaning, outlier detection (IQR), and scaling (Z-score/Min-Max).
* **`exercise_2_knn_metrics.m`**: kNN classification/regression, Bayesian optimization, and performance metrics (F1, ROC/AUC).
* **`exercise_3_svm_trees.m`**: Decision Trees and SVM implementation for astronomical and signal data.
* **`exercise_4_pendulum_ann.m`**: Neural Network training for non-linear pendulum period prediction.
* **`exercise_5_heat_transfer_main.m`**: Main execution for thermal resistance and interface temperature analysis.
* **`exercise_6_stochastic_sim.m`**: Monte Carlo simulation of nanoparticle movement and $O(N^p)$ complexity analysis.

### Helper Functions
* **`Heat_transfer_en.m`**: Core physics engine used by Exercise 5 (Must be in the same folder as the main script).

### Required Data Files
* **`Sensor_data.mat`** & **`physics_data_sample.csv`**: Used in Preprocessing (Ex 1).
* **`ionosphere.data`**: Radar data used in kNN metrics (Ex 2).
* **`magic04.data`**: Telescope data for classification (Ex 3).
* **`paper.xlsx`**: Path loss data for regression analysis (Ex 3).
