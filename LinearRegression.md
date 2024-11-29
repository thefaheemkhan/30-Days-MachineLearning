# Update Rule for Linear Regression and Logistic Regression

The **update rule** in both Linear Regression and Logistic Regression follows the framework of **Gradient Descent**. However, since the hypotheses and gradients differ, the updates are also slightly different.

---

## 1. Linear Regression

### Hypothesis:
\[
h_\theta(x) = \theta^T x
\]

### Cost Function (Mean Squared Error):
\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2
\]

### Gradient:
\[
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
\]

### Update Rule:
The parameters \( \theta \) are updated as:
\[
\theta_j := \theta_j - \alpha \cdot \frac{\partial J(\theta)}{\partial \theta_j}
\]

**Vectorized Update Rule:**
\[
\theta := \theta - \frac{\alpha}{m} X^T \left( X \theta - y \right)
\]

Where:
- \( X \): Feature matrix (\( m \times n \)).
- \( y \): Vector of actual values (\( m \times 1 \)).
- \( \theta \): Parameter vector (\( n \times 1 \)).

---

## 2. Logistic Regression

### Hypothesis (Sigmoid Function):
\[
h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}
\]

### Cost Function (Log-Loss):
\[
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]
\]

### Gradient:
\[
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
\]

### Update Rule:
The parameters \( \theta \) are updated as:
\[
\theta_j := \theta_j - \alpha \cdot \frac{\partial J(\theta)}{\partial \theta_j}
\]

**Vectorized Update Rule:**
\[
\theta := \theta - \frac{\alpha}{m} X^T \left( h_\theta(X) - y \right)
\]

Where:
- \( X \): Feature matrix (\( m \times n \)).
- \( h_\theta(X) \): Vector of predicted probabilities (\( m \times 1 \)), where \( h_\theta(X) = \frac{1}{1 + e^{-X \theta}} \).
- \( y \): Vector of actual values (\( m \times 1 \)).
- \( \theta \): Parameter vector (\( n \times 1 \)).

---

## Key Differences in Update Rules

| **Aspect**                | **Linear Regression**                                     | **Logistic Regression**                                 |
|---------------------------|----------------------------------------------------------|-------------------------------------------------------|
| **Hypothesis \( h_\theta(x) \)** | \( \theta^T x \) (linear)                             | \( \frac{1}{1 + e^{-\theta^T x}} \) (sigmoid)          |
| **Gradient Formula**      | \( \frac{1}{m} X^T (X \theta - y) \)                     | \( \frac{1}{m} X^T (h_\theta(X) - y) \)               |
| **Update Formula**        | \( \theta := \theta - \alpha \cdot \frac{\partial J}{\partial \theta} \) | \( \theta := \theta - \alpha \cdot \frac{\partial J}{\partial \theta} \) |
| **Nature of Updates**     | Linear relationship between \( \theta \) and \( x \).    | Non-linear updates due to sigmoid function.           |

---

## Summary

While the **update rule structure** is the same for Linear and Logistic Regression, the hypothesis and gradients make their behavior during Gradient Descent different:

- In **Linear Regression**, the updates are straightforward due to the linear relationship.
- In **Logistic Regression**, the sigmoid function introduces non-linearity, resulting in different model behavior.
