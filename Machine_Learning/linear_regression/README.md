# Multivariate Linear Regression From Scratch

## About

This project was built as part of my journey toward becoming an AI Engineer. The main goal was to understand how **multivariate linear regression works internally** by implementing the algorithm from scratch using Python and NumPy.

Instead of relying on machine learning libraries to train the model, I implemented the core calculations myself to better understand gradient descent, feature scaling, and model predictions.

## Concepts Learned

- **Multivariate Linear Regression** — using multiple input features to predict a target value.
- **Feature Scaling** — applying z-score normalization to help gradient descent converge more effectively.
- **Cost Function** — calculating the model's prediction error using the squared error cost function.
- **Gradient Descent** — updating weights and bias iteratively to minimize the cost.
- **Cost Tracking** — recording cost values across iterations to observe the learning process.
- **Making Predictions** — applying the trained model to new inputs using the same feature-scaling parameters from training.

## Technologies Used

- Python
- NumPy
- Matplotlib (for visualizing the cost, if included in the project)

## Model

The model uses the following equation:

\[
\hat{y} = w_1x_1 + w_2x_2 + b
\]

Where:
- \(\hat{y}\) is the predicted value.
- \(x_1, x_2\) are the input features.
- \(w_1, w_2\) are the model weights.
- \(b\) is the bias.

The model learns the weights and bias through gradient descent by minimizing the cost function.

## Dataset

A small example dataset is used to predict student marks from study hours and practice hours.

The dataset is intentionally simple and is intended for learning purposes, not for making real-world academic predictions.

## What I Practiced

- Implementing the mathematical operations behind a machine learning algorithm.
- Working with NumPy arrays and matrix operations.
- Understanding how weights and bias change during training.
- Using feature scaling consistently for training and new predictions.
- Observing how the cost changes during gradient descent.

## How to Run

1. Clone or download this repository.
2. Navigate to the project folder.
3. Install the required dependencies:

   ```bash
   pip install numpy matplotlib
   ```

4. Run the Python script:

   ```bash
   python main.py
   ```

   Replace `main.py` with the actual filename of your script.

## Limitations

- The dataset is very small and is only intended for experimentation.
- The input features in the example dataset are perfectly correlated, so their individual contributions cannot be distinguished reliably.
- The model's performance on this dataset does not indicate how well it would perform on real-world data.

## Purpose

This project is part of my ongoing AI and machine learning learning journey. My goal is to build a strong understanding of the fundamentals before moving on to more advanced models and machine learning projects.