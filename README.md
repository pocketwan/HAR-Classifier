# HAR-Classifier

This project explores the design and implementation of a deep learning pipeline to classify **human activities** using **raw time-series data** from the **MotionSense** dataset. While recognizing walking, sitting, or jogging from sensor data might seem straightforward, the reality presents several challenges, particularly around **sensor orientation**, **signal noise**, and **data segmentation**.

---

## Project Aim

To develop a machine learning classifier that accurately identifies user activities based on time-series data from the MotionSense dataset. Activities include:

- Walking
- Jogging
- Going upstairs
- Going downstairs
- Sitting
- Standing


## Solution Overview

This pipeline processes raw time-series sensor data using the following stages:

1. **Sliding window segmentation** to create overlapping time slices
2. **Standard scaling** to normalize feature ranges
3. **Label encoding** for categorical activity data
4. **Three-way data splitting** into training, validation, and test sets
5. **Stratified K-Fold Cross Validation** (5-fold)
6. **1D Convolutional Neural Network (CNN)** for activity classification
7. **Confusion matrix and training history plots** to evaluate performance



