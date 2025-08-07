# Event-by-Event SVD-Based Digit Classification

**Author:** Pramil Paudel  
**File:** `digit_training.m`  
**Description:**  
This MATLAB script implements an **event-by-event Singular Value Decomposition (SVD)** method for classifying handwritten digits (MNIST dataset).  
It incrementally updates **class-specific subspaces** only when a misclassification occurs, reducing unnecessary computation and memory usage.

---

## 📌 Features

- **Incremental Learning:** Updates SVD basis for a class only when a misclassification happens.
- **Batch-Wise Processing:** Evaluates both training and testing accuracy before each batch update.
- **Projection-Based Classification:** Classifies by projecting the sample onto each class subspace and using residual norms for decision-making.
- **Performance Tracking:**
  - Batch-wise training and testing accuracy.
  - Number of updates per batch.
  - Percentage of total training data used for updates.
- **ROC Curve Generation** for each digit class.

---

## 📂 Files Required

1. **Training Data:**  
   CSV file containing MNIST training data (labels in the first column, pixel values in the remaining columns).  
   Example: `train.csv`

2. **Test Data:**  
   CSV file containing MNIST test features (pixel values).  
   Example: `test.csv`

3. **Test Labels:**  
   CSV file containing corresponding labels for the test set.  
   Example: `mnist_submission.csv`
