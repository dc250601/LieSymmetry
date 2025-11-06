# Symmetry Discovery

## Introduction

Inductive Biasing a Neural Network is one of the most important aspect when it comes to designing a network architecture for downstream task. In many neural network architectures inductive baises are explored and embedded into systems to make them robust, accurate and data efficient. Symmetry a important form of inductive bias that is often used to design neural networks. Modern day machine learning (ML) explore symmetries in various different forms. Convolutional Neural Networks use translational symmetries, Graph Neural Networks use permutation symmetries while attention is know for permutation invariances. Symmetries are strong inductive biases that can be used to simplify learning and build explainable neural networks too.

The use of symmeties as inductive biases have shown excellent results when it comes to various downstream tasks as can be seen in [1](https://arxiv.org/abs/1703.06114),[2](https://arxiv.org/pdf/1905.11697),[3](https://arxiv.org/abs/2006.04780),[4](https://arxiv.org/abs/2201.08187) and [5](https://arxiv.org/abs/1612.08498).

![](https://codimd.web.cern.ch/uploads/upload_003d2c33bb299642cb337623a5eb4965.gif)

Even though symmetries offer better inductive biases, stronger expressive capacities and are more data efficient identifying the correct symmetry group that will help with the down stream task is something that is very difficult to determine. When wrong symmetry groups are used as inductive biases it ends up hurting the performance instead of providing any good. One good example to demonstrate this would be the MNIST dataset. For most part in the MNIST dataset the rotation symmetry group (SO(2)) helps with the classification performance but fails in the case of classifying between 6s and 9s where the SO(2) invariance is no longer true for the classification task. An SO(2) invariant neural network will perform very poorly (if not completly randomly) when met with such situations. Works like Augerino ([6](https://arxiv.org/pdf/2010.11882)) tries to address this by learning the extent of transformation (under symmetry) that is suitable before the performance starts going down.


![](https://codimd.web.cern.ch/uploads/upload_cc9356228bba95d2af7a8fe20c101f6f.png)
The above figure taken from [Augerino](https://arxiv.org/pdf/2010.11882) shows how rotating the 6 after a certain extent starts to hurt the performance, the paper tries to address this by learning the extent of transformation after which the loss starts to increase.

Even though Augerino manages to address the problem (of finding extent of symmetry transform) upto a certain extent the problem of indentifying the correct symmetry group suitable for a downstream task remains more or less unaddressed. 

To solve this very problem the LieGAN paper([7](https://arxiv.org/abs/2302.00236)) was introduced, the main intent of which was to use an Adversarial setup to discover the symmetries that lied within the given problem set. Once the symmetries are discovered the associated generators for these symmetry groups can be easily extracted which can be later used to build exact equivariant models that have strong inductive biases embedded within them.

This work was heavily inspired from the LieGAN setup but it differs from it significantly. The main contributions of this work can be summarised as below.
1. Removal of the Adversarial Game: LieGAN heavily relied on a adversarial setup for the task of symmetry discovery. The major problem with adversarial lossed are their tendency to collapse and instabilities with them. We have addressed this issue by removing the adversarial component thorugh reframing the problem and switching it with a more stable regression counterpart. This brings strong numerical stabilities and makes symmetry discovery possible for extremely large groups eg S0(10).
2. Modifying and simplifying the regularisation components of the LieGAN loss. The new regularisation terms produce cleaner results and forces stronger orthogonality on the solution.
3. Extending the task of symmetry discovery to complex systems where the symmetries are not expressible in linear form. In such cases a coordinate transformation has be discovered which brings the elements onto a coordinate space where symmetries can be easiliy discovered.  


## Definitions

**Dataset**: We define a dataset as an ordered pair $(X, y)$, where $X \in \mathbb{R}^m$ is the input vector and $y \in \mathbb{R}^n$ is the corresponding target or label vector. We denote the dataset as:

$$
\mathcal{D} = \{(X, y) \mid X \in \mathbb{R}^m,\ y \in \mathbb{R}^n\}
$$

**Oracle**: We define the Oracle function $\psi$ as a mapping from $\mathbb{R}^m$ to $\mathbb{R}^n$, such that:

$$
\psi(X) = y
$$
The orcale can both be learned from the ordered pair $(X, y)$ or simply embedded as a function where the analytic form of it is known.

**Encoder**: The encoder is a function $f_e$ such that it maps $X \in \mathbb{R}^m$ to a latent vector in $\mathbb{R}^\ell$, where $\ell$ is the latent dimension:
$$
f_e: \mathbb{R}^m \rightarrow \mathbb{R}^\ell
$$

**Decoder**: The decoder is a function $f_d$ such that it maps latent vectors from $\mathbb{R}^\ell$ back to the input space $\mathbb{R}^m$:
$$
f_d: \mathbb{R}^\ell \rightarrow \mathbb{R}^m
$$

**Latent**: The latent representation $Z$ of an input $X$ is given by:
$$
Z = f_e(X)
$$

**Reconstruction**: The reconstruction $\tilde{X}$ from a latent vector $Z$ is given by:
$$
\tilde{X} = f_d(Z)
$$

**MSE**: The mean squared error (MSE) is a function that takes two vectors $X = [X_1, \dots, X_N]$ and $\tilde{X} = [\tilde{X}_1, \dots, \tilde{X}_N]$ in $\mathbb{R}^{N \times m}$ and returns a scalar quantifying their squared difference. It is defined as:

$$
\text{MSE}(X, \tilde{X}) = \frac{1}{N} \sum_{i=1}^{N} \|X_i - \tilde{X}_i\|_2^2
$$

**Generator**: Let $\mathfrak{g}$ be the Lie algebra associated with a Lie group $G \subseteq \mathrm{GL}(d, \mathbb{R})$. The set $\{J_i\}$ denotes a basis of generators of $\mathfrak{g}$, where each $J_i \in \mathbb{R}^{d \times d}$.

**Matrix Exponential**: For a generator matrix $X \in \mathfrak{g}$, the exponential map is defined by the matrix exponential:

$$
\exp(X) = \sum_{n=0}^{\infty} \frac{X^n}{n!}
$$

This map takes elements from the Lie algebra $\mathfrak{g}$ to the Lie group $G$, i.e.,

$$
\exp: \mathfrak{g} \to G
$$

**Operator**: For a real parameter $\theta_i \in \mathbb{R}$ and generator $J_i$, we define the operator:

$$
\mathcal{O}_i = \exp(\theta_i J_i)
$$

**Symmetry Transform**: We define a symmetry transformation $f_s$ as the ordered product of exponentials over all generators, we follow this formulation primarily from the LieGAN paper:

$$
f_s(\{\theta_i\}) =  \exp(\sum_{i=1}^{k}\theta_i J_i)
$$

where $k$ is the number of generators used. Each transformation is parameterized by a real scalar $\theta_i \in \mathbb{R}$, which controls the strength of the generator $J_i$ in the exponential map. In prior work such as *Augerino*, these parameters $\theta_i$ are treated as learnable and optimized during training.
In our setting, we simplify the model by sampling the parameters $\theta_i$ from a uniform distribution over the interval $[-1, 1]$.

**Invariance**:  
We define a transformation $F: \mathbb{R}^m \to \mathbb{R}^m$ to be a **invariance** with respect to $\Psi: \mathbb{R}^m \to \mathbb{R}^n$ if:

$$
\Psi(F(x)) = \Psi(x)
$$


## Model Architecture

The architecure and losses for the symmetry discovery model is surprisingly simple given that it does not have any adversarial component associated to it. The symmetry discovery model for simple datasets where a infinitesimal group element can be written as a linear function $J_i$ is simply the symmetry transform $f_s$. The situation complicates when this is no longer true. The parameters of $f_s$ are estimated using gradient descent by minimising the following loss components.


**Closure Loss**:  
Given that we want to discover the transformations $f_s$ that are invariant to $\psi$ we define our first loss as:

$$
\mathcal{L}_{\text{closure}} = \text{MSE}(\psi(X), \psi(X_{\text{transform}}))
$$
where,
$$X_{\text{transform}} = \exp\left(\sum_{i=1}^{k} \theta_i J_i\right) \cdot X$$

Here, $\theta_i \in \mathbb{R}$ are sampled from a distribution (learned or uniform)

**Collapse Loss**:
The most trivial transformation that minimizes the closure loss $\mathcal{L}_{\text{closure}}$ is the **identity transformation**, i.e., when the input remains unchanged under the action of the transformation that is when all generator matrices are zero:
$$
J_i = 0_{d \times d}, \quad \forall i
$$
To prevent the system from trivially collapsing to Identity transformation we use following loss function:

We define an identity-regularization loss that penalizes when the transformation becomes trivially close to the identity. The loss is defined as:

$$
\mathcal{L}_{\text{collapse}} = \sum_{i=1}^{k} \left| \cos\left( \exp(\theta_i J_i) \cdot X,\ X \right) \right|
$$

Setting $\theta_i$ to $1$ and absorbing the component of $X$ in the equation this expression simplifies to a cosine similarity between the transformation operator and the identity matrix:

$$
\Longrightarrow\quad \mathcal{L}_{\text{collapse}} = \sum_{i=1}^{k} \left| \cos\left( \exp(\theta_i J_i),\ I_d \right) \right|
$$

The cosine is taken by flattening the transformation matrix.

**Orthogonality Loss** 
To promote diversity and reduce redundancy among generators $\{J_1, \dots, J_k\}$, we define a loss that penalizes their pairwise inner products after Frobenius normalization.

Let $\hat{J}_i = \frac{J_i}{\|J_i\|_F}$ denote the Frobenius-normalized generator. Then the loss is:

$$
\mathcal{L}_{\text{orth}} = \sum_{1 \leq i < j \leq k} \left( \langle \hat{J}_i, \hat{J}_j \rangle_F \right)^2
$$

where $\langle A, B \rangle_F = \operatorname{Tr}(A^\top B)$ denotes the Frobenius (Hilbert-Schmidt) inner product.

This encourages the generators to as different from each other as possible.

## Experiments

We sample sythetically generate the datasets for a few well know symmetry groups. 


### $\mathrm{SO}(N)$ — Special Orthogonal Group

The group $\mathrm{SO}(N)$ consists of all valid **rotation transformations** in $N$-dimensional Euclidean space. Formally,

$$
\mathrm{SO}(N) = \{ R \in \mathbb{R}^{N \times N} \mid R^\top R = I,\ \det(R) = 1 \}
$$

This group preserves the **Euclidean (L2) norm**, i.e., it conserves the radial distance from the origin. For any $x \in \mathbb{R}^N$ and $R \in \mathrm{SO}(N)$:

$$
\|Rx\|_2 = \|x\|_2
$$

Hence, $\mathrm{SO}(N)$ represents all **distance-preserving rotations** around the origin in $N$ dimensions.

We define an oracle function $\psi$ that returns the squared norm of an input vector $x$:

$$
\psi(x) = \sum_{i=1}^{N} x_i^2 = \|x\|_2^2
$$

Using our method we can easily discover the SO(N) group for the values of N upto 10, after which it becomes computationally expensive. All these generators are very close to their text-book counter parts and have very small degree of error within them.

### $\mathrm{SO}(3)$
![](https://codimd.web.cern.ch/uploads/upload_92ae8abaf07fba08b1418eef0373ca62.png)

### $\mathrm{SO}(4)$
![](https://codimd.web.cern.ch/uploads/upload_9b5f465f066c855fbd2d3b2e005c9814.png)

### $\mathrm{SO}(5)$
![](https://codimd.web.cern.ch/uploads/upload_db771e04a7e1a9e0c7e3f0d156c14a1b.png)

### $\mathrm{SO}(10)$
![](https://codimd.web.cern.ch/uploads/upload_930d830abba953a97b26c37871f7b9b0.png)


### $\mathrm{SO}^+(3,1)$ — Proper Orthocheronous Lorentz Group

The group $\mathrm{SO}^+(3,1)$ is the **proper orthochronous Lorentz group**, a subgroup of the full Lorentz group that consists of all **orientation- and time-preserving** linear transformations in $(3+1)$ dimensional Minkowski spacetime.

Formally, it is defined as:

$$
\mathrm{SO}^+(3,1) = \left\{ \Lambda \in \mathbb{R}^{4 \times 4} \ \middle|\ \Lambda^\top \eta \Lambda = \eta,\ \det(\Lambda) = 1,\ \Lambda^0_0 \geq 1 \right\}
$$

where $\eta$ is the Minkowski metric:

$$
\eta = \begin{bmatrix}
-1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

This group preserves the **Minkowski spacetime interval**:

$$
\psi(x) = -t^2 + x^2 + y^2 + z^2 = x^\top \eta x
$$

for any four-vector $x = (t, x, y, z)^\top \in \mathbb{R}^{4}$. That is,

$$
\psi(\Lambda x) = \psi(x), \quad \forall \Lambda \in \mathrm{SO}^+(3,1)
$$
![](https://codimd.web.cern.ch/uploads/upload_dc3a517097aae4770b9178086c83fdc0.png)


## Comparison with LieGAN

The motivation of this very work was to improve upon the idea of LieGAN. As seen from the last section we were quite successful in demonstrating the using regressional techniques can indeed work for symmetry discovery.

Here we compare the results of our implementation of Symmetry Regression (SymReg) with the LieGAN implementation interms of the closure loss or invariance loss.

Comparing the LieGAN (using their implementation) discovered symmetry generators of $\mathrm{SO}(3)$ with the ones generated with Symreg (ours) we see that LieGAN has a tendency to mix up the generator. Both the techniques were trained following similar conditions.

![](https://codimd.web.cern.ch/uploads/upload_16afe991f24d08be9cefc5cdb0376808.png)

Below we see the effects of evolving a single vector over similar number of steps (degree of augmentation) using both the generators discovered by LieGAN and SymReg. As expected we see that the radial component of the vector keeps diverging with time for the case of LieGAN which indicates towards presence of noise in the generators. The divergence in the case of SymReg is orders of magnitude smaller.

![](https://codimd.web.cern.ch/uploads/upload_434f3cae714b30161927138e1748d784.png)

We can also compare the mean error in closure loss for the generators discovered using the two methods and see that SymReg manages to minimise the error by a large margin when compared to LieGAN
![](https://codimd.web.cern.ch/uploads/upload_14bb4ec96aa51f27297ef73d9770b010.png)

## Hidden Symmetry

When we discussed the nature of our symmetry discovery function we stated that the infinitesimal symmetry transformation can be written as a linear function of the generators of the symmetry. When this case is not true cannot compute the symmetry function as simply the exponential map of linear combination of generators.

This problem was decribed in the works of Tegmark et al. [8](https://arxiv.org/abs/2109.09721). Such systems call for a coordinate transform that maps the original space to a new one where the symmetry generators are linearly dependent on the infinitesimal transformation. This makes the problem a join optisation task where both the symmetries are discovered along with the relevant coordinate transform. 

We now extend our formulation to support **latent-space symmetry discovery** by introducing an encoder–decoder architecture. The idea is to first map the input to a latent space where the symmetry group acts **linearly**, apply the transformation, and then decode back to the input space.

Let:
- $f_e: \mathbb{R}^m \to \mathbb{R}^\ell$ be the encoder,
- $f_d: \mathbb{R}^\ell \to \mathbb{R}^m$ be the decoder,
- $f_s = \exp\left(\sum_{i=1}^{k} \theta_i J_i\right)$ be the symmetry transformation acting in latent space.

Then the transformation proceeds as:

$$
Z_X = f_e(X)
$$

$$
X_{\text{transformed}} = f_d(f_s \cdot Z_X)
$$

The closure loss remains the same though the definition of transformation changes:

$$
\mathcal{L}_{\text{closure}} = \text{MSE}\left(\psi(X),\ \psi(X_{\text{transformed}})\right)
$$

To ensure bijectivity of both f_e and f_d we add the fourth component of loss into our overall optimisation conditions.

$$
\mathcal{L}_{\text{AE}} = \text{MSE}\left(X,\ f_d(f_e(X))\right)
$$

This acts as a regularizer that encourages the latent representation $Z_X = f_e(X)$ to retain sufficient information and prevents the latent space from collapsing to zero.

To demonstrate the efficacy of our proposed formulation, we evaluate its performance on a series of controlled toy examples. Specifically, we consider a scenario where the input space of a known $\mathrm{SO}(2)$ symmetry discovery problem is **nonlinearly transformed** by a predefined function.We then apply our encoder–decoder–based symmetry discovery model to this transformed space and assess whether it can successfully **recover the underlying $\mathrm{SO}(2)$ symmetry** that originally existed in the untransformed space.


#### Case I — Linear Scaling Perturbation: $X_{\text{new}} = 3X,\quad Y_{\text{new}} = Y$


<p align="center">
  <img src="https://codimd.web.cern.ch/uploads/upload_563d4c151961b443277ee940e521d5dd.png" width="48%" />
  <img src="https://codimd.web.cern.ch/uploads/upload_864e0b1aea6d3a52fb1904c59085ceb2.png" width="45%" />
</p>

The left scatter plot shows the original and the transformed datapoints. The plot of the right has a scatter plot that shows coordinate transform that the encoder is embedding the latents to. The black arrows shows the invariant symmetry transformations associated to the blue points on the decoder space.


#### Case II — Exponential Perturbation: $X_{\text{new}} = e^{X},\quad Y_{\text{new}} = e^{Y}$

<p align="center">
  <img src="https://codimd.web.cern.ch/uploads/upload_b0ec112445f23686204726ca72d79a44.png" width="48%" />
  <img src="https://codimd.web.cern.ch/uploads/upload_a38104de0678b7c92b012769729d7edc.png" width="45%" />
</p>


#### Case III — Mixed Nonlinear Perturbation: $X_{\text{new}} = \operatorname{sign}(X) \cdot \sqrt{|X|},\quad Y_{\text{new}} = e^{Y}$

<p align="center">
  <img src="https://codimd.web.cern.ch/uploads/upload_89f3f5536882a41e9cf21747213940a9.png" width="48%" />
  <img src="https://codimd.web.cern.ch/uploads/upload_0df67a92f82fe774174eef3f53de449f.png" width="45%" />
</p>

As evident from the above diagrams and plots that the system works well both in terms of dicovering symmetries and finding the hidden coordinate transformations.

## Conclusion and Future Directions

In this work, we proposed a stable, regression-based framework for symmetry discovery. Our approach directly learns the infinitesimal generators of a Lie group. It was demonstrated that this method can be used to recover known symmetry groups such as $\mathrm{SO}(N)$ and $\mathrm{SO}^+(3,1)$, and can be extended to systems where the symmetry is not linearly expressible in the original coordinates. For such cases, we introduced a learnable coordinate transformation via an encoder–decoder architecture to reveal latent representations where the symmetry acts linearly.

Future work will involve applying this framework to real-world datasets, scaling it to larger and more complex groups, and integrating the discovered symmetries into downstream tasks such as classification, regression and anomaly detection systems.

## References

1. https://arxiv.org/abs/1703.06114
2. https://arxiv.org/pdf/1905.11697
3. https://arxiv.org/abs/2006.04780
4. https://arxiv.org/abs/2201.08187
5. https://arxiv.org/abs/1612.08498
6. https://arxiv.org/pdf/2010.11882
7. https://arxiv.org/abs/2302.00236
8. https://arxiv.org/abs/2109.09721