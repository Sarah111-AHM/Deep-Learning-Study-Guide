# Chapter 2: Linear Algebra

## Foundation for Deep Learning
Linear Algebra is the backbone of Deep Learning.  
Every deep learning algorithm uses matrices to represent data and model parameters.

## Essential Mathematical Prerequisites
Before diving into deep learning, you should understand:
- Scalars, Matrices, and Tensors
- Matrix Operations
- Norms and Special Matrices
- Matrix Decompositions
- Principal Component Analysis (PCA)

## Outline of the Chapter
1. Why Every Deep Learning Algorithm Speaks Matrices
2. Scalar, Matrix & Tensor
3. Matrix Operations
4. House Price Prediction Example
5. Norms & Special Matrices
6. Matrix Decompositions
7. Principal Component Analysis (PCA)

## Tip
Practice creating and manipulating matrices in Python using **NumPy**.  
Try small examples like predicting house prices with simple linear algebra.
---
# linear Algebra is the language of ML 
في Deep Learning كل البيانات وكل العمليات الحسابية تتحول إلى مصفوفات Matrices أو تينسورات Tensors ؟! لان الحواسيب تتعامل مع الأرقام بشكل أفضل عندما تكون في شكل شبكات ثنائية الأبعاد أو متعددة الأبعاد 

# Dataset = Matrix
![](https://drive.google.com/uc?export=view&id=1xYlFbGUkWyb8lJH2br96CkuD1oJQ9M0Q)
## Every Dataset is a Matrix
كل Dataset يتحول لمصفوفة لانه يسهل التعامل مع البيانات جميعها دفعة واحدة ويتيح العمليات الرياضية بكفاءة
---

### Example Matrix X

| Sample | Feature 1 | Feature 2 | Feature 3 |
|--------|-----------|-----------|-----------|
| 1      | x11       | x12       | x13       |
| 2      | x21       | x22       | x23       |
| 3      | x31       | x32       | x33       |

**Structure:**

- **Rows** = Individual samples / data points
- **Columns** = Features / attributes
- `X ∈ ℝ^(n×m)` → n samples, m features

---

### Why Matrices?

- Process **all data at once**
  بدل حساب كل عينة على حدة، يمكن للـ DL model التعامل مع كل البيانات دفعة واحدة باستخدام مصفوفة
- Leverage **GPU acceleration**
GPUs مصممة لمعالجة العمليات على المصفوفات بسرعة عالية، وبالتالي التدريب يصبح أسرع بكثير
- Apply **mathematical operations**
كل العمليات الرياضية مثل: الجمع، الضرب، الضرب العنصري، ضرب المصفوفات، يمكن تطبيقها مباشرة على المصفوفة
# Machine Learning = Matrix Operations

---

## Linear Regression

\[
\hat{y} = X W + \hat{h}
\]

- **X**: Data matrix (n samples × m features)  
- **W**: Weight vector (m × 1)  
- **b**: Bias (scalar)  
- **ŷ**: Predictions (n × 1)  

> Pure matrix multiplication!

---

## PCA (Dimensionality Reduction)

\[
X_{\text{new}} = X V
\]

- **X**: Original data matrix  
- **V**: Eigenvectors (principal components)  
- **X_new**: Transformed data  

> Eigenvalue decomposition!

---

## Neural Networks

\[
h = a(Wx + b)
\]

- **W**: Weight matrix  
- **x**: Input vector  
- **a**: Activation function (non-linear)  
- **h**: Hidden layer output  

> Stack of matrix operations!

---

## Distance / Similarity

- **Norm**: Vector length  
- **x, y**: Data points (vectors)  
- **d**: Distance  

> Vector operations!

---

## Deep Learning Pipeline

From Data → Predictions:

1. **Raw Data** → Matrix X  
2. **Preprocess** → Scale / Transform  
3. **Model** → Matrix Operations  
4. **Predictions** → Vector ŷ

---

## Preprocessing

- **Scaling**: \(X = (X - \mu) / \alpha\)  
- **Normalization**: \(X' = X / ||X||\)  
- **Whitening**: \(X^{\text{whitened}} = XV\)

---

## Training

- **Forward pass**: \(\hat{y} = f(XW)\)  
- **Loss**: \(L = ||y - \hat{y}||^2\)  
- **Gradient**: \(\nabla W = X^T \Delta\)  
- **Update**: \(W = W - \alpha \nabla W\)

> Inference:  
> Compute \(\hat{y} = XW + b\) → Output prediction!

---

## Why Deep Learning Needs Linear Algebra

### GPU Acceleration

- GPUs have thousands of cores → parallel matrix operations  
- Example: \(1000 \times 1000\) matrix × \(1000 \times 1000\) → 1 billion multiplications  
  - **CPU**: ~10 seconds  
  - **GPU**: ~0.01 seconds → 1000× faster!

### Massive Scale

| Model      | Parameters |
|------------|------------|
| ResNet-50  | 25M        |
| BERT       | 340M       |
| GPT-3      | 1.758B     |
| GPT-4      | 1.7T       |

> Each parameter = a matrix element → training involves billions of multiplications and millions of gradient updates

---

> All in linear algebra!
# The Bottom Line

**Deep Learning = Optimized Linear Algebra at Scale**

**Key Insight:**  
Understanding linear algebra = Understanding how deep learning actually works
# Scalars

**Definition:** Single number

**Characteristics:**
- Just one number
- Can be integers, real numbers, rational numbers, etc.
- Usually written in *italics* with lowercase names: `a, n, x`

**Examples:**
- \(s \in \mathbb{R}\) → real-valued scalar  
- \(n \in \mathbb{N}\) → natural number scalar
# Vectors

**Definition:** A vector is a 1-D array of numbers

**Characteristics:**
- Can be real, binary, integer, etc.
- Elements: \(x_1, x_2, \dots, x_j\)
- Vector space: \(x \in \mathbb{R}^n\)
# Matrices

**Structure:** 2-D array of numbers

**Characteristics:**
- Represented by uppercase bold letters (e.g., `A`)  
- Shape: \(A \in \mathbb{R}^{m \times n}\)  
- Height = \(m\), Width = \(n\)

**Indexing:**
- Element: \(A_{i,j}\) (italic, not bold)  
- Row \(i\): \(A_{i,:}\)  
- Column \(j\): \(A_{:,j}\)
# Tensors

**Definition:** Multi-dimensional arrays (more than 2 axes)

**Characteristics:**
- Regular grid with variable number of axes  
- Represented with special typeface: `𝓐`  
- Element access: \(A_{i,j,k}\)  

**Visualize:** Think of a 3D cube, 4D hypercube, or n-dimensional structure

**Rule of Thumb:**  
You can't improve what you don't understand
# Matrix Operations

---

## 1. Transpose Operation

**Definition:** Mirror image / flip across the main diagonal  

**Notation:** \(A^T\)  

\[
(A^T)_{i,j} = A_{j,i}
\]

**Vector transpose:** Converts row → column or column → row  

**Example:**

\[
A = 
\begin{bmatrix}
a & b & e \\
d & e & t
\end{bmatrix}_{2 \times 3} 
\quad \Rightarrow \quad
A^T = 
\begin{bmatrix}
a & d \\
b & e \\
e & t
\end{bmatrix}_{3 \times 2}
\]

---

## 2. Addition

**Definition:** Element-wise addition of matrices of the same shape  

\[
C = A + B, \quad C_{i,j} = A_{i,j} + B_{i,j}
\]

**Broadcasting:** Adding a vector to each row of a matrix  

\[
C = A + b, \quad C_{i,j} = A_{i,j} + b_j
\]

---

## 3. Scalar Multiplication

**Definition:** Multiply every element of a matrix by a scalar  

\[
D = \alpha \cdot B
\]

---

## 4. Matrix Multiplication

**Definition:** Number of columns in \(A\) must equal number of rows in \(B\)  

\[
C = AB, \quad C_{i,j} = \sum_k A_{i,k} B_{k,j}
\]

**Dimensions:**
- \(A: m \times n\)  
- \(B: n \times p\)  
- \(C: m \times p\)

**Example:**
- \(A\) shape: (3,2)  
- \(B\) shape: (2,4)  
- \(C = AB\) shape: (3,4)

**Usage:** Linear transformations, solving systems of equations

---

## 5. Hadamard Product (Element-wise Multiplication)

**Definition:** Multiply matrices element by element  

\[
C = A \odot B, \quad C_{i,j} = A_{i,j} \cdot B_{i,j}
\]

**Usage:** Neural networks, masking operations
# System of Linear Equations

**Definition:** A set of linear equations involving the same set of variables.

**General Form:**

\[
\begin{cases}
a_{11}x_1 + a_{12}x_2 + \dots + a_{1n}x_n = b_1 \\
a_{21}x_1 + a_{22}x_2 + \dots + a_{2n}x_n = b_2 \\
\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \dots + a_{mn}x_n = b_m
\end{cases}
\]

**Matrix Form:**

\[
AX = B
\]

- \(A\) = coefficient matrix (\(m \times n\))  
- \(X\) = variable vector (\(n \times 1\))  
- \(B\) = constants vector (\(m \times 1\))

**Solution Methods:**
- Direct: Gaussian elimination, LU decomposition  
- Iterative: Gradient descent, Jacobi, Gauss-Seidel
# System of Linear Equations

**Matrix Form:**  
\[
AX = b
\]

- **A**: Known matrix (\(m \times n\))  
- **X**: Unknown vector (\(n \times 1\)) → Solve for this!  
- **b**: Known vector (\(m \times 1\)), \(b \in \mathbb{R}^m\)  

**Expanded Form:**  
\[
A_{i,1} x_1 + A_{i,2} x_2 + \dots + A_{i,n} x_n = b_i, \quad i = 1,\dots,m
\]

---

## Identity Matrix

**Definition:** Neutral element for multiplication (scalar analogue: 1)  

- **Notation:** \(I_n\)  
- **Dimension:** \(n \times n\)  
- **Structure:** Main diagonal = 1, all other entries = 0  

**Example (3×3):**  
\[
I_3 =
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
\]

---

## Matrix Inverse

**Definition:** The inverse of a square matrix \(A\) is denoted \(A^{-1}\) and satisfies:  

\[
A^{-1} A = I_n
\]

> Practical note: In software, we usually avoid computing \(A^{-1}\) directly; specialized algorithms (e.g., LU decomposition) give better numerical precision.

---

## Solving Systems of Linear Equations

A linear system can have:  
1. **No solution**  
2. **Many solutions**  
3. **Exactly one solution** → The matrix multiplication is invertible  

**Example:**  
\[
5x + 1 = 2x + 10 \quad \Rightarrow \quad 3x = 9 \quad \Rightarrow \quad x = 3
\]

---

## Matrix Invertibility Conditions

- Only square matrices (\(n \times n\)) can be invertible  
- Determinant \(\det(A) \neq 0\)  
- Columns (and rows) must be linearly independent
# Matrix Invertibility Conditions

**Definition:** A matrix \(A\) is invertible (\(A^{-1}\) exists) if and only if the linear system \(AX = b\) has exactly **one solution for any vector \(b\)**.

**Conditions for Invertibility:**

1. **Square Matrix:** Number of rows = number of columns (\(m = n\))  
2. **Linear Independence:** All columns (or rows) must be linearly independent  
3. **Singular Matrix:** A square matrix with linearly dependent columns is singular → not invertible  
4. **Inverse Property:** For square invertible matrices, the left inverse equals the right inverse
# Problem: Predict House Prices

**Goal:** Find the relationship between house features and price using **Linear Regression**.

---

## Features (Input)

- Size (sq ft)  
- Number of Bedrooms  
- Age (years)  
- Distance to city center  

**Target (Output):**  
- House Price (in thousands of dollars)

---

## Linear Algebra Translation

- **Matrix A** → Feature values for each house  
- **Vector x** → Weights (unknowns to learn)  
- **Vector b** → Prices (target)

---

## Sample Data (4 Houses)

| House | Size | Bedrooms | Age | Distance | Price |
|-------|------|----------|-----|---------|-------|
| 1     | 2000 | 3        | 10  | 5       | 295   |
| 2     | 1500 | 2        | 15  | 8       | 202.5 |
| 3     | 2500 | 4        | 5   | 3       | 382.5 |
| 4     | 1800 | 3        | 12  | 10      | 245   |

**Observations:**  
- Larger size → Higher price  
- More bedrooms → Higher price  
- Older age → Lower price  
- Further distance → Lower price

---

## Matrix Formulation

\[
A x = b
\]

- **Matrix A (Features):**

\[
A = 
\begin{bmatrix}
2000 & 3 & 10 & 5 \\
1500 & 2 & 15 & 8 \\
2500 & 4 & 5  & 3 \\
1800 & 3 & 12 & 10
\end{bmatrix}_{4 \times 4}
\]

- **Vector x (Weights / Unknowns):**  

\[
x = 
\begin{bmatrix}
w_{\text{size}} \\
w_{\text{bedrooms}} \\
w_{\text{age}} \\
w_{\text{distance}}
\end{bmatrix}_{4 \times 1}
\]

- **Vector b (Prices):**  

\[
b = 
\begin{bmatrix}
295 \\
202.5 \\
382.5 \\
245
\end{bmatrix}_{4 \times 1}
\]

**Interpretation:**  
- Each row of \(A\) → Features of a house  
- Each column of \(A\) → One feature across all houses  

---

## Expanded System of Equations

\[
\begin{cases}
2000 w_1 + 3 w_2 + 10 w_3 + 5 w_4 = 295 \\
1500 w_1 + 2 w_2 + 15 w_3 + 8 w_4 = 202.5 \\
2500 w_1 + 4 w_2 + 5 w_3 + 3 w_4 = 382.5 \\
1800 w_1 + 3 w_2 + 12 w_3 + 10 w_4 = 245
\end{cases}
\]

- **System Properties:** 4 equations, 4 unknowns → square system → should have unique solution

---

## Solution Using Matrix Inverse

\[
x = A^{-1} b
\]

> Important Note: In practice, for larger systems we **avoid computing \(A^{-1}\) directly**.  
> Numerical methods used:  
> - LU Decomposition  
> - QR Decomposition  
> - Gaussian Elimination  
> - Iterative methods
# Computing x = A⁻¹b (Numerical Solution)

**Using computational tools** (Python/NumPy, MATLAB, etc.) to solve the system \(Ax = b\).

**Output (Weights):**

| Feature       | Weight |
|---------------|--------|
| Size          | 0.150 $/sq ft |
| Bedrooms      | 15,000 $ per bedroom |
| Age           | -2,500 $ per year |
| Distance      | -5,000 $ per km |

**Interpretation:**

- Each extra sq ft adds **$150** to the price  
- Each bedroom adds **$15,000**  
- Each year of age reduces price by **$2,500**  
- Each km from the city reduces price by **$5,000**

---

## Verification

**Test House:**

- Size: 1700 sq ft  
- Bedrooms: 4  
- Age: 12 years  
- Distance: 6 km  
- Actual Price: $270,000

**Predicted Price:**

\[
\text{Price} = 0.150 \cdot 1700 + 15,000 \cdot 4 - 2,500 \cdot 12 - 5,000 \cdot 6 = 266,600
\]

**Error:** $3,400 → Excellent fit!

---

## Making New Predictions

**New House Features:**

- Size: 2200 sq ft  
- Bedrooms: 3  
- Age: 8 years  
- Distance: 6 km

**Predicted Price:**

\[
\text{Price} = 0.150 \cdot 2200 + 15,000 \cdot 3 - 2,500 \cdot 8 - 5,000 \cdot 6 = 325,000
\]

**Success!** Model can now predict the price of any house given its features using the learned weights.

---

## Real-World Machine Learning Considerations

1. **Overdetermined systems** (more equations than unknowns → \(m > n\))  
   - Use **Least Squares** solution  

2. **Underdetermined systems** (fewer equations than unknowns → \(m < n\))  
   - Infinite solutions possible  

3. **Linearly dependent columns** → Matrix is singular  
   - Use **pseudoinverse** instead of exact inverse  

**In practice:**  
- Usually many more samples than features (e.g., 1000 houses & 4 features)  
- System is overdetermined → no exact solution exists, approximate solution via Least Squares

---

## Why Did This Work?

**Necessary Conditions Met:**

1. **Square Matrix:** \(A\) is 4×4 (same number of equations and unknowns)  
2. **Inverse Exists:** Unique solution guaranteed  
3. **Linearly Independent Columns:** Each feature provides unique information, matrix is non-singular
# Norms in Machine Learning

## What is a Norm?

**Definition:**  
A norm is a function that maps vectors to **non-negative numbers** and measures their “length” or “size”.

**Three Required Properties:**

1. **Non-negativity:** \(\|x\| \ge 0\), and \(\|x\| = 0 \iff x = 0\)  
2. **Absolute scalability:** \(\|\alpha x\| = |\alpha| \|x\|\)  
3. **Triangle inequality:** \(\|x + y\| \le \|x\| + \|y\|\)

---

## Common Norms

### 1. L2 Norm (Euclidean Norm)

\[
\|x\|_2 = \sqrt{x_1^2 + x_2^2 + \dots + x_n^2}
\]

- Most common norm  
- Measures straight-line distance  
- Key property: sensitive to outliers  

---

### 2. L1 Norm (Manhattan Distance)

\[
\|x\|_1 = |x_1| + |x_2| + \dots + |x_n|
\]

- Measures distance along axes (like grid paths)  
- Common in machine learning for **sparse solutions**  
- Less sensitive to outliers  

---

### 3. L∞ Norm (Max Norm)

\[
\|x\|_\infty = \max(|x_1|, |x_2|, \dots, |x_n|)
\]

- Measures the largest magnitude among vector components  
- Useful in **robust optimization** and bounded analyses  

---

## Matrix Norms & Dot Product

- Matrix norms generalize vector norms  
- **Dot product** interpretable in terms of angle between vectors:  
  - Positive → similar direction  
  - Zero → orthogonal  
  - Negative → opposite direction  

---

## Use Cases in Machine Learning

- Comparing distances between points (Euclidean vs Manhattan)  
- Regularization in regression (L1 → Lasso, L2 → Ridge)  
- Measuring magnitudes of gradients in optimization
# Special Kinds of Matrices and Vectors

---

## 1. Symmetric Matrices

**Definition:**  
A matrix \(A\) is symmetric if it equals its transpose:  
\[
A = A^T
\]

**Where They Arise:**  
- Distance matrices  
- Covariance matrices in statistics

**Example:**  
\[
A = 
\begin{bmatrix}
1 & 2 & 3 \\
2 & 4 & 5 \\
3 & 5 & 6
\end{bmatrix}
\]

---

## 2. Diagonal Matrices

**Definition:**  
A square matrix where all off-diagonal elements are zero:  
\[
D = 
\begin{bmatrix}
d_1 & 0 & 0 \\
0 & d_2 & 0 \\
0 & 0 & d_3
\end{bmatrix}
\]

- Simplifies matrix operations like inversion and powers  
- Often used in eigenvalue decomposition

---

## 3. Unit & Orthogonal Vectors

### Unit Vector

- A vector with **length 1**:  
\[
\|v\| = 1
\]

### Orthogonal Vectors

- Two vectors \(u\) and \(v\) are orthogonal if their dot product is zero:  
\[
u \cdot v = 0
\]

### Orthonormal Vectors

- Vectors that are both **unit length** and **mutually orthogonal**  
- Often used as basis vectors in vector spaces

---

## 4. Applications

- **Cosine Similarity:** Measures angle between vectors, often used in recommender systems  
- **Collaborative Filtering:** Uses vector operations and similarity measures for recommendations
# Orthogonal Matrices

**Definition:**  
A square matrix \(Q\) is orthogonal if its transpose equals its inverse:  

\[
Q^T Q = Q Q^T = I
\]

**Key Properties:**
- Preserves vector lengths: \(\|Qx\| = \|x\|\)  
- Preserves angles between vectors  
- Determinant = ±1  

**Uses in Machine Learning:**
- Multi-layer Perceptrons (MLP) weight initialization  
- Principal Component Analysis (PCA)  
- Numerical stability in matrix computations  

---

# Matrix Decompositions

**Definition:** Factorizing a matrix into simpler matrices to simplify computations  

**Common Decompositions:**

1. **LU Decomposition:** \(A = LU\) (Lower × Upper triangular matrices)  
2. **QR Decomposition:** \(A = QR\) (Orthogonal × Upper triangular)  
3. **Cholesky Decomposition:** \(A = Q A Q^T\) (for symmetric positive definite matrices)  
4. **Singular Value Decomposition (SVD):** \(A = U \Sigma V^T\)  

**Core Concept:**  
Decompositions allow us to represent a complex matrix as **interconnected factors**, simplifying solutions, inversion, or eigenvalue computations.

---

# Eigenvectors and Eigenvalues

**Definition:**  
For a square matrix \(A\), a non-zero vector \(v\) is an **eigenvector** if:  

\[
A v = \lambda v
\]

where \(\lambda\) is the corresponding **eigenvalue**.

**Key Properties:**
- Scale invariance: Only the direction of \(v\) matters, not its length  
- Singular matrices have zero eigenvalues  
- Used in dimensionality reduction (PCA), stability analysis, and more

**Eigen Decomposition Formula:**  
\[
A = V \Lambda V^{-1}
\]

- \(V\) = matrix of eigenvectors  
- \(\Lambda\) = diagonal matrix of eigenvalues
# Eigen Decomposition

**Formula:**  
\[
A V = V \, \text{diag}(\lambda)
\]

Where:  
- \(V\) = matrix of eigenvectors (\(v_1, v_2, \dots\))  
- \(\text{diag}(\lambda)\) = diagonal matrix of eigenvalues (\(\lambda_1, \lambda_2, \dots\))  

**Visualization:**  
Matrix \(A\) stretches space in the direction of each eigenvector by its corresponding eigenvalue.

---

## How to Compute Eigenvalues and Eigenvectors

1. **Find the Characteristic Equation:**  
\[
\det(A - \lambda I) = 0
\]  
Where \(I\) is the identity matrix. This gives the **characteristic polynomial**.

2. **Solve for Eigenvalues:**  
Find the roots of the characteristic polynomial. These roots are the eigenvalues \(\lambda_i\).

3. **Find Eigenvectors:**  
For each eigenvalue \(\lambda_i\), solve:  
\[
(A - \lambda_i I)v_i = 0
\]  
The non-zero solutions \(v_i\) are the eigenvectors corresponding to \(\lambda_i\).

---

## Example

Find eigenvalues and eigenvectors of  

\[
A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}
\]

**Step 1: Characteristic Equation**  

\[
A - \lambda I = 
\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} - \lambda \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} =
\begin{bmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{bmatrix}
\]

\[
\det(A - \lambda I) = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = 0
\]

**Step 2: Solve for Eigenvalues**  

\[
\lambda_1 = 1, \quad \lambda_2 = 3
\]

**Step 3: Solve for Eigenvectors**  

For \(\lambda_1 = 1\):  
\[
(A - 1 I)v_1 = 0 \quad \Rightarrow \quad v_1 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}
\]

For \(\lambda_2 = 3\):  
\[
(A - 3 I)v_2 = 0 \quad \Rightarrow \quad v_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
\]

---

## Application: Google PageRank

- PageRank uses **eigenvalues and eigenvectors** to rank webpages.  
- The principal eigenvector of the **link matrix** gives the steady-state distribution of page importance.  
- This is the mathematical core behind Google’s search algorithm success.
# The Search Ranking Problem: Google PageRank

## Popularity Contest Analogy

Imagine trying to find the most popular person in a school:  
- Not just counting friends  
- Popular people have popular friends  
- A's opinion matters more if A is popular themselves  

**Key Idea:** Popularity of a person (or webpage) is **recursive** — it depends on the popularity of others who reference them.  
This is exactly how **PageRank** works for web pages.

---

## The Challenge

- Billions of web pages to rank  
- Simple keyword counting fails  
- Spammers can manipulate keywords  
- Need an **objective quality measure**  
- Must **scale** to the entire web  

**Problem:** How to measure page importance objectively?  
> A page is important if important pages link to it!  

This recursive definition naturally leads to **eigenvectors and eigenvalues**.

---

## How PageRank Works

1. Represent the web as a **link matrix** \(M\), where:  
\[
M_{ij} = \text{probability of a user navigating from page } i \text{ to page } j
\]

2. Solve the **eigenvector equation**:  
\[
v = M v
\]  
where \(v\) is the PageRank vector (steady-state distribution of page importance)

3. The **largest eigenvalue (λ=1)** corresponds to the principal eigenvector, which ranks pages.

---

## Example (Illustrative)

| Page | Linked From | Probability (Mij) |
|------|------------|-----------------|
| 1    | 2,3        | 0.5, 0.5        |
| 2    | 1,3        | 0.5, 0.5        |
| 3    | 1,2        | 0.5, 0.5        |
| 4    | 5          | 1               |
| 5    | 1          | 1               |

- Compute eigenvector of \(M\)  
- The resulting vector gives the **PageRank scores**  
- Pages are ranked according to these scores

---

## Impact & Significance

- PageRank provides an **objective measure of page importance**  
- Basis for **Google's search algorithm**  
- Uses **linear algebra at scale**, connecting **eigenvectors** to a real-world problem
# PageRank: Impact and Significance

## Business and Historical Impact

- **PageRank was the rocket fuel for Google's success**  
- Before PageRank: simple keyword matching → weak search engines  
- PageRank algorithm: powerful, eigenvector-based ranking  
- Modern search: Machine Learning models like BERT use PageRank as **one signal**  

**Key Business Milestones:**

- Google founded: 1998 based on PageRank  
- Market domination: better results → more users  
- Advertising revolution: AdWords built on search  
- Trillion-dollar company: all started with eigenvectors

---

## Timeline of Google PageRank

| Year | Event |
|------|-------|
| 1996 | PageRank concept developed at Stanford |
| 1998 | Original Google: ~25M web pages, matrix size 25M×25M, computation took several days |
| 2000s | Google growth, larger datasets |
| 2010s | Advanced search and ML signals |
| 2024 | Modern Google: 200+ billion web pages, matrix size ~200B×200B, computed continuously using massive distributed systems |

**Modern Google Ranking:**  
- PageRank is still used, but only **one of 200+ ranking factors**

---

## PageRank Beyond Web Search

The PageRank algorithm works for **any network or graph**:

- **Social Networks:** rank influential users (Twitter/X, LinkedIn)  
- **Biology:** protein interaction networks, disease spread (super-spreaders)  
- **Power Grids:** failure analysis, resource allocation  

**Nodes = entities, Edges = connections**

---

# Singular Value Decomposition (SVD)

**Advantage:** Works for any matrix, not just square ones.

For an \(m \times n\) matrix \(A\):

\[
A = U \, \Sigma \, V^T
\]

Where:  
- \(U\) = \(m \times m\) orthogonal matrix (left-singular vectors)  
- \(\Sigma\) = \(m \times n\) diagonal matrix (singular values)  
- \(V\) = \(n \times n\) orthogonal matrix (right-singular vectors)

**Connection to Eigenvectors:**  
- Left-singular vectors = eigenvectors of \(A A^T\)  
- Right-singular vectors = eigenvectors of \(A^T A\)  
- Singular values = square roots of eigenvalues

SVD is widely used in:  
- Dimensionality reduction (PCA)  
- Recommender systems  
- Signal processing and data compression

# Eigen Decomposition vs Singular Value Decomposition (SVD)

| Feature | Eigen Decomposition | Singular Value Decomposition (SVD) |
|---------|------------------|-----------------------------------|
| Applicability | Only for square matrices | Works for any matrix |
| Output | Reveals stretching directions along eigenvectors | Generalizes matrix inversion and reveals principal components |
| Use Cases | Optimization, stability analysis | Data science, dimensionality reduction, PCA, recommender systems |
| Key Insight | Both decompositions help understand **geometric transformations** represented by matrices | Both decompositions help understand **geometric transformations** represented by matrices |

---

# Principal Component Analysis (PCA) Explained

**Linear Algebra Made Simple: Understanding PCA with Intuition, Analogies & Visuals**

---

## The Story: A Data Science Problem

Imagine a **House Hunting Dataset**:  
- 10,000 houses  
- 100 features each: square footage, number of rooms, number of bathrooms, age, distance to school/mall, lot size, garage size, ceiling height, window count, door count, floor type, wall color, roof type, ... (and 86 more!)  

**The Problem:**  
- 100 features = 100 dimensions → impossible to visualize (humans see max 3D)  
- Many features are correlated (e.g., "# of bedrooms" and "# of bathrooms" both relate to house size)  
- Models can overfit and be inefficient

---

## The Question

Can we reduce 100 features to **10–20 features** without losing important information?  

**Answer:** Yes! Using **PCA**

---

## Core Concept

- PCA finds the **most important patterns** in your data  
- Creates new **super-features** (principal components) that capture the essence of all 100 original features  
- Reduces dimensionality while preserving the most relevant information
# Principal Component Analysis (PCA) – Core Concept

## Variance = Information

**Photography Analogy:**  
Imagine photographing a car:  

- **Side view:** shows length, doors, windows → **lots of detail**  
- **Front view:** shows width, headlights → **some detail**  
- **Top view:** shows length & width → **less detail**  

> The side view has **maximum variance**, shows the most differences between cars, and contains the **most information**.  

---

## What is Variance?

- Measures how **spread out** your data is  
- More spread → more variation → more information  

**PCA's Goal:**  
- Find directions in the data with **maximum variance**  
- Analogy: PCA chooses the best "camera angles" to capture the most information about your data

---

## Principal Components

1. **1st Principal Component (PC1):** Direction with **most variance**  
2. **2nd Principal Component (PC2):** Direction with **second most variance** (perpendicular to PC1)  
3. **3rd, 4th, 5th… Principal Components:** Each captures the **remaining variance** in orthogonal directions  

> PCA transforms the original high-dimensional data into a new coordinate system where the axes (principal components) are **ordered by the amount of variance they capture**.

---

## PCA Computation (Linear Algebra Approach)

- Step-by-step mathematical computation of principal components  
- Uses **covariance matrix** of data and **eigen decomposition / SVD**  
- Principal components = **eigenvectors of covariance matrix**  
- Variance captured = **corresponding eigenvalues**

# PCA: Mathematical Foundation

## The Architecture Analogy

Building PCA is like constructing a building:  

- **Data Matrix (X):** Raw materials (bricks, steel, concrete)  
- **Covariance Matrix:** Blueprint (shows relationships between features)  
- **Eigen Decomposition:** Structural engineering (finds strongest directions)  
- **Principal Components:** Finished building (organized, efficient structure)  

---

## Key Mathematical Concepts

- **Matrix Multiplication:** Combining transformations  
- **Covariance:** Measuring relationships between features  
- **Eigenvectors:** Special directions that do not change under transformation  
- **Eigenvalues:** Importance (variance) along each direction  
- **Orthogonality:** Perpendicular directions  
- **Variance:** Spread of data along directions  

**Goal:**  
\[
Y = X W
\]  
Where:  
- \(X\) = Original data (n×d matrix)  
- \(W\) = Transformation matrix (d×k)  
- \(Y\) = Transformed data (n×k)  
- \(k < d\) → dimensionality reduction

---

## Steps for Principal Component Analysis

1. **Data Standardization**  
2. **Covariance Matrix Computation**  
3. **Eigen Decomposition**  
4. **Sort & Select Principal Components**  
5. **Transform Data**

---

## Step 1: Data Standardization

**The Fair Competition Analogy:**  
- Imagine a competition with different scoring systems:  
  - Math test: 0–100 points  
  - Essay: 0–20 points  
  - Lab work: 0–5 points  
- Without standardization, math scores dominate simply because they have bigger numbers.  

**Standardization Formula:**  
\[
X_{\text{std}} = \frac{X - \mu}{\sigma}
\]  
Where, for each feature:  
- Subtract the mean (\(\mu\))  
- Divide by standard deviation (\(\sigma\))  

**Result:**  
- Mean = 0, Variance = 1  
- Prevents scale domination  
- Ensures fair feature contribution  
- Required before computing covariance
# PCA: Steps 2–4

## Step 2: Covariance Matrix

**The Friendship Network Analogy:**  
- The covariance matrix is like a **social network diagram** showing relationships between features:  
  - **Diagonal elements:** Variance of each feature (how each person interacts with themselves)  
  - **Off-diagonal elements:** Covariance between pairs of features (how people interact)  
    - Positive values → "friends who do things together"  
    - Negative values → "people who avoid each other"  

**Covariance Matrix Formula:**  
\[
\text{Cov}(X) = \frac{1}{n-1} X^T X
\]  
Where:  
- \(X\) = Standardized data matrix (n×d)  
- \(X^T\) = Transpose of X  
- \(n\) = Number of samples  

**Example: Before vs After Standardization**

| Feature | Original Data | Standardized Data |
|---------|---------------|-----------------|
| Height  | 180, 165, 175 | 1.2, -0.8, 0.4 |
| Weight  | 75, 68, 72    | 1.1, -0.7, 0.3 |

- Standardization ensures all features are **on the same scale** with **mean 0**  

---

## Step 3: Eigen Decomposition

- Solve the **eigen equation**:  
\[
C v = \lambda v
\]  
Where:  
- \(C\) = Covariance matrix  
- \(v\) = Eigenvector (direction of a principal component)  
- \(\lambda\) = Eigenvalue (importance / variance along this direction)  

**Process:**  
1. Compute characteristic equation: \(\text{det}(C - \lambda I) = 0\)  
2. Solve for eigenvalues (\(\lambda_1, \lambda_2, \dots\))  
3. For each eigenvalue, solve \((C - \lambda I) v = 0\) to get eigenvectors  
4. Normalize eigenvectors (unit length)  

> Eigenvectors give the directions of principal components; eigenvalues give the variance captured along each component.

---

## Step 4: Sort & Select Components

- Sort eigenvectors by **descending eigenvalues** (largest variance first)  
- Select the top **k principal components** for dimensionality reduction  
- Transform original data onto the new basis:  
\[
Y = X W
\]  
Where:  
- \(W\) = Matrix of selected eigenvectors (d×k)  
- \(Y\) = Reduced-dimensional representation (n×k)
# PCA: Steps 4 & 5 – Sort, Select Components & Transform Data

## Step 4: Sort & Select Components

**Sorting Process:**  
1. Sort eigenvalues in **descending order**:  
\[
\lambda_1 \geq \lambda_2 \geq \lambda_3 \geq \dots \geq \lambda_n
\]  
2. Rearrange corresponding eigenvectors accordingly  
3. Calculate **variance explained** by each principal component  
4. Choose top **k components** that capture a desired percentage of variance (e.g., 95%)  

**Component Selection:**  
- Retain components that capture the majority of the information  
- Reduces dimensionality while preserving data variance  

---

## Step 5: Transform Data

**Transformation Formula:**  
\[
Y = X W
\]  
Where:  
- \(X\) = Standardized data (n×d)  
- \(W\) = Matrix of selected eigenvectors (d×k)  
- \(Y\) = Transformed data (n×k)  
- \(k < d\) → reduced dimensionality  

---

## Complete Numerical Example

**Sample Dataset:** 3 samples, 2 features (\(X_1, X_2\))  

**Step 1: Standardize**  

| Feature | Mean (\(\mu\)) | Std (\(\sigma\)) | Standardized Data |
|---------|----------------|-----------------|-----------------|
| X1      | 4              | 2               | 0, 1, -1        |
| X2      | 5              | 2               | 0, -1, 1        |

---

**Step 2: Covariance Matrix**  

\[
\text{Cov}(X) = \frac{1}{n-1} X^T X = 
\begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}
\]

---

**Step 3: Eigen Decomposition**  

- Eigenvalues: \(\lambda_1 = 3\), \(\lambda_2 = 1\)  
- Eigenvectors (corresponding to \(\lambda_1, \lambda_2\)):

\[
v_1 = \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix}, \quad
v_2 = \begin{bmatrix} 1/\sqrt{2} \\ -1/\sqrt{2} \end{bmatrix}
\]

---

**Step 4 & 5: Transform**  

- Keep **PC1 only** (\(\lambda_1 = 3\)) → captures **all significant variance**  
- Transform data:  
\[
Y = X W = X v_1
\]  
- Result: Reduced from 2D → 1D while keeping 100% variance

---

## Final Takeaways

- PCA reduces **dimensionality** efficiently  
- Retains **most variance/information**  
- Steps involve: **Standardization → Covariance → Eigen Decomposition → Sort → Transform**  
- Applicable in **data visualization, compression, and feature extraction**

