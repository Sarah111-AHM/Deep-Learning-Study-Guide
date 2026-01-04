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
