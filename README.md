# AIC354: Machine Learning Fundamentals 🤖

Welcome to my repository for **AIC354: Machine Learning Fundamentals**, a 3(2,1) credit-hour course mapping core algorithmic theory to practical computing implementations. This repository serves as a centralized portfolio documenting my weekly laboratory tasks, data preprocessing workflows, and end-to-end machine learning pipelines.

---

## 📌 Course Overview & Structure
* **Course Code:** AIC354
* **Department:** Computer Science, COMSATS University Islamabad
* **Focus Areas:** Supervised & Ensemble Learning, Unsupervised Learning, Optimization & Model Selection, and Reinforcement Learning.

### 🎯 Applied Course Learning Outcomes (Lab Focus)
All practical workflows in this repository are designed to fulfill university-level performance objectives:
* **CLO-5 (Applying):** Implement supervised, unsupervised, and reinforcement learning algorithms using specialized modern machine learning frameworks and tools.
* **CLO-6 (Creating):** Architect and develop comprehensive machine learning pipelines spanning data inspection to model evaluation.

---

## 📁 Repository Directory Matrix

```text
├── Lab_Task_02_Text_Preprocessing/    # CLO-6 Text Cleaning & Duplicate Filtration
│   ├── basic_graph.py                 # Initial data manipulation scripts
│   └── text_preprocessing.py          # Lowercasing, regex URL/mention removal scripts
├── Lab_Task_03_Dataset_Analysis/      # CLO-5 Descriptive Statistics & Exploration
│   ├── iris.csv                       # Local target dataset matrix
│   └── Iris.py                        # Tabular ingestion, info summary, and feature math
└── README.md                          # Repository profile documentation
```

---

## 🗺️ Engineering & Implementation Roadmap

The labs in this repository trace the theoretical benchmarks outlined in the institutional plan:

| Phase / Unit | Milestone Focus | Covered Algorithms & Methodologies |
| :--- | :--- | :--- |
| **Phase 1: Supervised Basics** | Foundational Pipelines | Text Data Preprocessing, Localized Ingestion, Baseline Metric Discovery |
| **Phase 2: Core Supervised** | Algorithmic Classification & Regressions | K-Nearest Neighbors (KNN), Decision Trees, Naive Bayes, Linear & Logistic Regression |
| **Phase 3: Deep Networks** | Optimization Frameworks | Perceptrons, Artificial Neural Networks (ANN), Backpropagation, Gradient Descent |
| **Phase 4: Optimization** | Regularization & Validations | Bias-Variance Evaluation, Regularization (L1, L2), Confusion Matrix Assessment |
| **Phase 5: Unsupervised** | Structural Grouping & Reduction | K-means, Hierarchical Clustering, Principal Component Analysis (PCA) |
| **Phase 6: Reinforcement** | Complex Behavioral Agents | Markov Decision Processes (MDP), Temporal Difference Learning, Deep Q-Networks (DQN) |

---

## 🛠️ Environment Configuration & Tools

To isolate local tracking dependencies safely, code executions run in a localized virtual workspace:

```bash
# 1. Initialize the Python virtual environment module
python -m venv myenv

# 2. Engage active runtime configurations (PowerShell example)
.\myenv\Scripts\Activate.ps1

# 3. Component deployments
pip install pandas scikit-learn matplotlib
```

---

## 📚 Core References
* **Textbook:** *Machine Learning*, Alpaydin, E., The MIT Press, 2021.
* **Reference:** *Practical Machine Learning: A Beginner's Guide with Ethical Insights*, Nyamawe et al., CRC Press, 2025.
* **Reference:** *Reinforcement Learning: An Introduction*, Sutton & Barto, MIT Press, 2018.
