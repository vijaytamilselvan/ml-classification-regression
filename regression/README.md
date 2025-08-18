# 📊 Linear Regression from Scratch (Gradient Descent)

This project implements **Linear Regression** from scratch using **NumPy** and **Gradient Descent** — without relying on scikit-learn’s built-in regression.  
It demonstrates both the **mathematics** behind linear regression and the **Python implementation**.

---

## 🔍 Project Overview
- Dataset is **synthetic** (ad spend on TV & Radio vs Sales).
- Implements **Linear Regression with Gradient Descent**.
- Shows how the **loss decreases over epochs**.
- Visualizes the **training loss curve**.

---

## 🧮 Mathematical Concepts

### 1. Hypothesis Function  

We want to model the relationship:  

$$
\hat{y} = w_0 + w_1 x_1 + w_2 x_2
$$  

Where:  
- $w_0$ = bias (intercept)  
- $w_1, w_2$ = weights (coefficients for TV & Radio)  
- $x_1, x_2$ = input features  

---

### 2. Loss Function (MSE)  

We use **Mean Squared Error (MSE):**  

$$
J(w) = \frac{1}{n} \sum_{i=1}^n (\hat{y}^{(i)} - y^{(i)})^2
$$  

This measures how far our predictions $\hat{y}$ are from actual values $y$.  

---


### 3. Gradient Descent  

To minimize the loss, update weights iteratively:  

$$
w := w - \alpha \cdot \frac{\partial J(w)}{\partial w}
$$  

Where:  
- $\alpha$ = learning rate  

Gradient with respect to weights:  

$$
\frac{\partial J}{\partial w} = \frac{2}{n} X^T(\hat{y} - y)
$$  

This ensures the weights move in the direction of **steepest descent** to reduce loss.

---

## 🖥️ Implementation Steps

1. **Generate Synthetic Dataset**  
   - Features: TV & Radio advertisement spend  
   - Target: Sales (linear function with noise)  

2. **Preprocess Data**  
   - Add bias column to $X$  
   - Initialize weights randomly  

3. **Train Model with Gradient Descent**  
   - Forward pass → predict $\hat{y}$  
   - Compute loss (MSE)  
   - Backpropagate gradient  
   - Update weights  

4. **Track Training**  
   - Print weights & loss every 100 epochs  
   - Plot loss curve  

---

## 📈 Results

- Final Weights (Bias, TV, Radio) approximate the true relationship:  

$$
Sales \approx 3 \times TV + 1.5 \times Radio + 5
$$  

- Loss decreases smoothly with epochs.  
- Model learns the underlying function correctly.  

📊 **Example Loss Curve:**  

*(Insert matplotlib loss plot here if available)*  

---

## 🚀 How to Run

```bash
git clone https://github.com/<your-username>/linear-regression-scratch.git
cd linear-regression-scratch
python linear_regression.py
