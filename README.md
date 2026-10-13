# 🏡 Automated Residential Property Valuation & Mortgage Risk Terminal

An end-to-end Machine Learning Automated Valuation Model (AVM) designed to estimate fair-market residential property prices and streamline mortgage Loan-to-Value (LTV) underwriting decisions.

Built for the **Machine Learning Laboratory (Experiment 10)** at Pune Vidyarthi Griha's College of Engineering and Technology (PVGCOET).

---

## 📌 Project Overview
Manual real-estate appraisals cost financial lenders between $350 and $600 per survey and introduce subjective valuation bias. This system provides an automated, explainable valuation engine that benchmarks regularized linear methods, bagging ensembles, and gradient boosting algorithms on real-world demographic and structural property data.

### Key Capabilities:
- **Multi-Model Benchmark:** Evaluates Ridge Regression, Random Forest, and XGBoost Regressor across $R^2$, MAE, RMSE, and MAPE metrics.
- **Interactive Web Terminal:** FastAPI backend serving a responsive Tailwind CSS dashboard with live appraisal calculators and explainability drivers.
- **Underwriting Impact:** Quantifies operational savings by automating approvals for over 70% of standard loans within a $\pm 10\%$ LTV risk tolerance.

---

## 📊 Benchmark Results

| Model Architecture | Paradigm | $R^2$ Score | MAE ($) | RMSE ($) | MAPE (%) | Engineering Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ridge Regression** | Regularized Linear | 0.5758 | $52,620 | $70,350 | 26.8% | Baseline |
| **Random Forest** | Bagging Ensemble | 0.7912 | $32,800 | $49,200 | 16.4% | Benchmark |
| **XGBoost Regressor** | Gradient Boosting | **0.8354** | **$28,400** | **$43,800** | **13.9%** | **Selected Model** |

---

## 🏗️ Architecture & Directory Structure

```text
├── plots/                           # Generated evaluation artifacts
│   ├── 1_price_distribution.png     # Target distribution & skewness
│   ├── 2_income_vs_price.png        # Feature correlation scatter
│   ├── 3_correlation_heatmap.png    # Covariance matrix heatmap
│   └── 4_actual_vs_predicted.png    # XGBoost regression fit curve
├── static/
│   └── index.html                   # Tailwind CSS frontend dashboard
├── app.py                           # FastAPI server & inference endpoints
├── housing_pipeline.py              # ML data processing, training, & metrics
├── REPORT.md                        # 14-section formal academic technical report
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation


🚀 Quick Start Guide

1. Clone the RepositoryBashgit clone [https://github.com/](https://github.com/)<YOUR_GITHUB_USERNAME>/<REPO_NAME>.git
cd <REPO_NAME>


2. Install DependenciesBashpip install -r requirements.txt


3. Run the ML PipelineGenerates the plots, trains all three regression models, and outputs the underwriting financial matrix:Bashpython housing_pipeline.py


4. Launch the Interactive DashboardBashpython -m uvicorn app:app --reload --port 8000
Open http://localhost:8000 in your browser to interact with the live valuation terminal.



📈 Engineering Interpretation
$R^2$ Variance Explained: The selected XGBoost model explains 83.54% of property price variation across test block groups.Operational Savings: By automating 71.4% of property valuations within acceptable mortgage appraisal tolerances, a financial lender processing 1,000 monthly applications saves an estimated $321,300 in on-site manual inspection fees.



📜 LicenseThis project is open-source under the MIT License.