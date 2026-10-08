# Machine Learning — Course Projects

Completed graduate coursework for an M.Sc. in Artificial Intelligence: game-state values, imbalanced ensembles and supervised models under real-world data imperfections.

## Overview

Original materials identify **Machine Learning, Spring 2026**. HW1 report names **Dr. Sattar Hashemi** as instructor. University/department remain unverified: **[Add university]**, **[Add department]**. Degree context is supplied by the portfolio brief.

Three projects retain original code, notebooks, reports, data, weights and results. Documentation follows static implementation evidence and saved artifacts; it does not certify correctness or reproduce experiments.

## Course projects

| Project / topic | Main techniques | Languages/tools | Documentation |
| --- | --- | --- | --- |
| [Homework 1: Linear Value-Function Learning for Games](1/) | Linear value approximation, TD-style learning, self-play | Python, NumPy, Pygame | [README](1/README.generated.md) |
| [Homework 2: Ensemble Learning for Imbalanced Classification](2/) | Hellinger trees, undersampled bagging, AdaBoost.M1, SMOTE | Python, pandas, scikit-learn, Matplotlib | [README](2/README.md) |
| [Homework 3: Kernel and Non-Kernel Learning Under Real-World Data Imperfections](3/) | Data engineering, linear/KNN/tree models, SVM, KRR, KPCA, kernel geometry | Python, Jupyter, NumPy, pandas, SciPy | [README](3/README.md) |

## Skills demonstrated

- Feature-based value learning, TD-style updates and self-play.
- Imbalanced classification, Hellinger trees, bagging, boosting and SMOTE.
- Train-only preprocessing, feature engineering, grouped and temporal splits.
- Custom linear, neighbor, tree and kernel models; metrics, geometry and failure/group analysis.
- Python, Jupyter, NumPy, pandas, SciPy, scikit-learn, Matplotlib, seaborn and Pygame.

## Repository structure

```text
1/                         Game agents, assignment, reports, weights
  README.generated.md      Original README preserved
2/                         Ensemble experiments: code/, results/, figures/
  README.md
3/                         Kernel study: src/, processed/, artifacts/, cache/
  ML_Homework_Complete.ipynb  README.md
docs/                      Static GitHub Pages portfolio
.portfolio/                Original-file inventory and validator
PUBLICATION_REVIEW.md       Privacy/ownership findings
PAGES_SETUP.md              Hosting instructions
PORTFOLIO_GENERATION_REPORT.md
```

## Getting started

```bash
git clone https://github.com/Mana-Peiravian/machine-learning-projects.git
cd machine-learning-projects
```

Read each project README before installing anything. The projects have different environments; no global dependency file is assumed. Commands are inferred and untested. Reproduce in isolated copies because scripts write outputs. HW3 raw datasets are absent; processed tables and saved results remain inspectable.

## GitHub Pages

Portfolio: [https://mana-peiravian.github.io/machine-learning-projects/](https://mana-peiravian.github.io/machine-learning-projects/). This remote-derived URL is **pending publication**, not verified live. See [Pages setup](PAGES_SETUP.md). No backend, install or build is needed. Update [main.js](docs/assets/js/main.js) configuration and HTML fallback links if publishing elsewhere.

## Academic integrity and usage

Completed coursework is intended for portfolio, reference and educational use. Current students must follow institutional academic-integrity policies. Assignment descriptions may remain their authors' or institution's intellectual property. Reports and datasets need separate permission checks. Read [publication review](PUBLICATION_REVIEW.md) and [license notice](LICENSE-NOTICE.md) before release.

## Author

Reports name **Mana Peiravian**; HW1/HW2 also name **Sina Sabooki** as collaborator. HW3 individual authorship is not inferred. Confirm collaborator permission. Optional public contact/LinkedIn: **[Add approved public contact]**, **[Add LinkedIn URL]**. Student numbers are omitted.
