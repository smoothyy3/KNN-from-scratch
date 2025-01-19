# KNN-from-scratch
This project demonstrates the implementation of the k-Nearest Neighbors (k-NN) algorithm entirely from scratch in Python. Unlike most implementations, this project does not use any machine learning imports nor basic libraries like math, numpy, or pandas. The goal is to provide an extremely fundamental, step-by-step approach to understanding this classic machine learning algorithm.

- **Visual Explanation of k-NN**:  
  The following diagram illustrates how the k-NN algorithm works:  
  <img src="./assets/knn.png" alt="KNN Formula" width="400">

where **M**<sub>k</sub>(**q**) is the prediction of the model **M** for the query **q** given the parameter of
the model k; *levels(t)* is the set of levels in the domain of the target feature, and *l* is an
element of this set; *i* iterates over the instances **d**<sub>i</sub> in increasing distance from the query
**q**; t<sub>i</sub> is the value of the target feature for instance **d**<sub>i</sub>; and δ(t<sub>i</sub>,l) is the Kronecker delta
function, which takes two parameters and returns 1 if they are equal and 0 otherwise.

## Features

### **1. Pure Python Implementation**
  - No external libraries or helper functions are used. Everything is built from the ground up.
    
### **2. Multiple Distance Metrics**
  - The project supports the following distance metrics, implemented manually:
    - **Minkowski Distance**  
      <img src="./assets/minkowski_distance.png" alt="Minkowski Distance Formula" width="400">
    - **Euclidean Distance**  
      <img src="./assets/euclidean_distance.png" alt="Euclidean Distance Formula" width="400">
    - **Manhattan Distance**  
      <img src="./assets/manhatten_distance.png" alt="Manhattan Distance Formula" width="400">

  ### **3. Scalers**
  To preprocess data effectively, two scaling methods are implemented from scratch:

  - **Standard Scaler**  
  Standardizes features by removing the mean and scaling to unit variance:  
    <img src="./assets/Formel-z-score.png" alt="Z-Score Formula" width="400" height="150">

- **Min-Max Scaler**  
  Scales features to a fixed range, typically \([0, 1]\):  
    <img src="./assets/minmaxformula.png" alt="Min-Max Formula" width="400">

  ## Usage
1. **Dataset Preparation**
   - Generate a dataset with training and test data.

2. **Data Scaling (Optional)**
   - Use the `ScaleData` class to scale features using either `standard` or `minmax` scaling.

3. **Train and Predict**
   - Fit the k-NN model with training data and predict for test data.

## To-Do
- [ ] Add support for additional distance metrics (e.g., cosine similarity).
- [ ] Add accuracy for classification.
- [ ] Add regression.
- [ ] Add visualization tools for decision boundaries.
      
