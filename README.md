# GameFlow: Adversarial Game Dynamics and Optimal-Transport Adaptation

A reproducible mini-project studying **two-player game dynamics**, **distributional trajectories in generative adversarial learning**, and a practical **optimal-transport (OT) image-classification demonstration**.

The project is organized into two independent Jupyter notebooks. **Both notebooks are included exactly as supplied, with their code and saved outputs unchanged.** No refactoring, retraining, additional architectures, or new experimental results have been introduced in this repository.

## What the project demonstrates

1. **A solvable two-player zero-sum game.** Simultaneous gradient descent--ascent can converge, oscillate, cycle, or diverge even though the game has an equilibrium. Eigenvalues of the discrete update matrix provide an exact stability criterion.
2. **A NumPy GAN on real 8 × 8 handwritten-digit images.** The generator and discriminator are trained using the original minimax objective. Changes in their relative learning rates alter the quality of generated images and the geometry of the training trajectory. Empirical Wasserstein-1 distance is evaluated via optimal assignment.
3. **OT-Shift: an operational classification demo.** An RBF-SVM trained once on clean digits is evaluated on synthetically darkened images. A label-free, batch-level, one-dimensional intensity transport can correct this particular distribution shift without retraining the classifier.

**Important:** The GAN experiment is a two-player game. The OT-Shift classifier is a *separate practical extension*, **not** a GAN or a second Nash-equilibrium experiment. They share the mathematical language of probability distributions and optimal transport.

## Repository contents

```text
GameFlow-Optimal-Transport/
├── GameTheory_project(2).ipynb   # 28×28 WGAN-GP (optional), scalar game, 8×8 NumPy GAN
├── Live_update(1).ipynb          # independent OT-Shift classifier and live prediction examples
├── README.md
├── requirements.txt              # core experiments: NumPy, SciPy, scikit-learn, matplotlib, Jupyter
├── requirements-mnist.txt        # optional PyTorch / torchvision dependencies for 28×28 experiment
└── .gitignore
```

The notebooks already contain saved plots and outputs. The archive does **not** include generated `outputs/` training artifacts, downloaded MNIST data, or pretrained GAN checkpoints.

## Installation

Use Python 3.10 or newer and create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the lightweight dependencies and start Jupyter:

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

**GPU is not required** for OT-Shift, the scalar game, or the 8×8 NumPy GAN. The latter is nevertheless computationally expensive because it repeatedly solves 512×512 discrete assignments.

**Only if you want to run the separate 28×28 MNIST WGAN-GP experiment:**

```bash
python -m pip install -r requirements-mnist.txt
```

That experiment uses PyTorch and torchvision, downloads MNIST on first run, and benefits from a CUDA-capable GPU. Install PyTorch and torchvision using the matching builds recommended for your platform if this general installation does not work.

## Quick demonstration: OT-Shift (recommended starting point)

Open **`Live_update(1).ipynb`** and run its cells **sequentially from the top**. This notebook is independent and does not require running the GAN notebook first.

Its main steps are:

1. Load scikit-learn's built-in 8×8 grayscale digit dataset (1,797 samples); make stratified training, validation, and test partitions (1,168 / 269 / 360 images).
2. Fit an RBF-kernel SVM (`C=10`, `gamma='scale'`) on the clean training partition.
3. Calibrate a Wasserstein-1 shift threshold using 120 clean, unlabeled validation batches of 128 images each (99th percentile of the observed distances).
4. Simulate exposure reduction by multiplying image intensities by 0.35; compare direct predictions with unlabeled batch-level intensity quantile matching.
5. Visualize corrected predictions. The last two cells also call the live demonstration on specific test indices.

For a live example, run this in a new cell **after the notebook's earlier cells have been executed**:

```python
ot_shift_demo(index=10, exposure=0.20)
```

`index` must identify one of the 360 held-out images (`0`–`359`); `exposure` must be between `0.20` and `1.00`. The function displays the original, darkened and corrected digit, followed by the predicted labels and shift distance. In the saved output for this particular example, the darkened digit is predicted as 4 and the corrected digit as 6 (true digit 6).

### Recorded OT-Shift result

| Condition | Accuracy on the 360 held-out digits |
| --- | ---: |
| Clean image input | **99.4%** |
| Input scaled to 35% exposure, without correction | **73.1%** |
| Same frozen classifier, after OT correction | **99.4%** |

For this setup, the original notebook reports intensity Wasserstein distance **0.3883 before** and **0.0010 after** adaptation; the detector threshold is about **0.0090**. It also reports **97 originally incorrect dark-image predictions corrected** by OT. The final score is 358/360; correction can also change previously correct predictions, so do not interpret the rescued count as a guarantee for every image.

**Scope:** These are the notebook's saved results on a controlled *global, monotonic brightness-scaling* corruption with a fixed, unlabeled test batch. The same test batch is used to estimate the transport map, which makes this **transductive, batch-level adaptation**. This is not evidence of general domain adaptation, adversarial robustness, or state-of-the-art image-recognition performance.

## Game theory and image GAN: notebook guide

Open **`GameTheory_project(2).ipynb`**. The code cells are organized as follows (after the notebook's heading):

| Code cell | Experiment | Status / how to use |
| --- | --- | --- |
| 1 | **28×28 MNIST WGAN-GP** with sliced Wasserstein-1 metrics, PyTorch | Optional; starts training and downloads MNIST. Its stored output stops partway through a run, so no completed quantitative result is asserted here. Requires `requirements-mnist.txt`. |
| 2 | **Scalar zero-sum game** with closed-form Wasserstein distance and a learning-rate stability atlas | Complete and light to run. Uses core dependencies only. |
| 3 | **8×8 NumPy GAN**, four training regimes, three seeds, exact empirical assignment Wasserstein distance, 48-point learning-rate sweep | Complete but computationally expensive. Uses core dependencies only. Running this cell starts training by default. |

**Do not select “Run All”** if you only intend to reproduce the completed scalar-game and 8×8 image-GAN experiments: the first code cell would also start the optional 28×28 MNIST training.

### Scalar game

The scalar payoff is

$$
V(g,d)=gd-\frac{\gamma}{2}d^2, \quad \gamma=1.2.
$$

The generator minimizes and discriminator maximizes this payoff. Their simultaneous updates are

$$
\begin{bmatrix}g_{t+1}\\d_{t+1}\end{bmatrix}=
\begin{bmatrix}1&-\eta_g\\\eta_d&1-\gamma\eta_d\end{bmatrix}
\begin{bmatrix}g_t\\d_t\end{bmatrix}.
$$

The origin is the unique Nash equilibrium, and asymptotic convergence of the discrete linear dynamics requires spectral radius $\rho(A)<1$. For positive rates, the stability criterion is

$$
\eta_g<\gamma, \qquad \eta_d(2\gamma-\eta_g)<4.
$$

The generator's output law is $\mathcal N(g_t,1)$ and the reference is $\mathcal N(0,1)$, so their **population** Wasserstein-1 distance is exactly $|g_t|$. The notebook compares stable, damped-oscillatory, marginally oscillatory and unstable updates over 200 steps, and visualizes the phase portraits, distribution densities, eigenspectra, and learning-rate sweep.

### 8×8 image GAN

The image experiment uses **scikit-learn digits**, not MNIST. Its generator is `16 → 64 → 64` (noise vector → hidden layer → pixels); its discriminator is `64 → 64 → 1`. Backpropagation is implemented manually in NumPy. Training follows the original minimax GAN objective with simultaneous stochastic updates.

The default complete run comprises:

- Four selected generator/discriminator learning-rate regimes, **3 seeds each**, **3,000 updates per run**;
- A separate **8 × 6 = 48** learning-rate grid sweep, **one seed** and **2,000 updates per configuration**;
- Evaluations of generated and reference empirical measures with **512 samples** and 64-dimensional Euclidean assignment cost;
- Saved images, convergence/phase plots, path measurements and a CSV table in `outputs/`.

The stored notebook output reports these representative results (mean ± standard deviation over the three seeds):

| Game regime | Final empirical $W_1$ | Observed path length |
| --- | ---: | ---: |
| Balanced | 1.697 ± 0.025 | 197.6 ± 12.3 |
| Generator-fast | 1.759 ± 0.072 | 457.1 ± 33.6 |
| Discriminator-dominant | 3.304 ± 0.058 | approximately 0.1 |
| Large learning rates | 5.093 ± 0.737 | 4.5 ± 0.4 |

The generator-fast configuration traverses a substantially longer measured distributional path than the balanced configuration despite similar final $W_1$. Path length measures *distributional movement*, not computational FLOPs or training time. The distance is **exact for the two fixed empirical measures**, not the unknown underlying population distributions.

When run, this code creates `outputs/results.pkl`, `outputs/results_table.csv`, and PNG figures. The notebook's `--replot` mode only works after an `outputs/results.pkl` file has been generated.

**Execution caveat:** The NumPy GAN cell uses `multiprocessing.Pool(2)`. In some environments, notably Windows/Jupyter kernels using the `spawn` multiprocessing method, notebook-defined worker functions may not be importable by subprocesses. Run this unmodified notebook in a Linux Jupyter environment (or a compatible environment with process forking) if you encounter that issue. It is a runtime/environment constraint, not a missing dependency.

## Methodological distinctions and limitations

- **Scalar vs. neural game:** The spectral-radius stability condition is analytical for the 2D scalar game; it is **not** a proven stability condition for the full neural GAN.
- **OT diagnostics vs. OT intervention:** The image GAN uses optimal transport to *measure* how generated distributions evolve; OT-Shift uses a 1D transport map to *modify* input pixel intensities before classification.
- **Image-space metric:** The NumPy GAN's optimal assignment uses Euclidean distance between flattened 8×8 images; it is not a semantic image-quality benchmark.
- **Incomplete optional run:** The 28×28 WGAN-GP cell is included exactly as given but has no complete saved result. It uses *sliced* Wasserstein distance, not the same exact empirical assignment metric as the 8×8 GAN.
- **No novel SOTA claim:** This repository demonstrates established game-theoretic and optimal-transport ideas with a small practical proof of concept; it does not establish new state-of-the-art performance.

## Publish on GitHub (optional)

This ZIP is a **GitHub-ready local repository layout**; it has **not** been published to an online GitHub account. After unpacking it, create an empty repository on GitHub and run the following commands from this folder:

```bash
git init
git add .
git commit -m "Add original GameFlow notebooks and documentation"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin main
```

Replace `YOUR-USERNAME/YOUR-REPO` with your actual repository destination. No remote destination or public/private visibility is assumed by this package.

## Contributors

- **Vraj Motiwala** — U24ECO57
- **Vishal Murugadhas** — U24EC007
- **Kaustubh Kapil** — U24EC049

This repository is distributed as project source material. No software license has been specified.
