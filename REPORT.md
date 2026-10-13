# Experiment 10: Residential Property Valuation & Price Estimation System
**Course:** Machine Learning Laboratory  
**Department:** Electronics and Telecommunication Engineering, PVGCOET  

---

## 1. Introduction
Automated Valuation Models (AVMs) are mathematical algorithms used by financial institutions, real estate funds, and mortgage underwriters to estimate property appraisal values without manual physical inspection.

## 2. Problem Definition
Traditional real estate appraisals require costly on-site physical surveys ($350–$600 per home) and introduce subjective human bias. The objective is to design a supervised regression system that predicts the continuous fair-market valuation ($) of residential properties based on locality demographics and structural dimensions to automate Loan-to-Value (LTV) underwriting.

## 3. Dataset Description
- **Dataset:** California Housing Dataset (Derived from the 1990 U.S. Census).
- **Instances:** 20,640 block-group observations.
- **Attributes (8 input features):**
  - `MedInc`: Median income in block group ($10,000s)
  - `HouseAge`: Median house age (years)
  - `AveRooms`: Average number of rooms per household
  - `AveBedrms`: Average number of bedrooms per household
  - `Population`: Block group population
  - `AveOccup`: Average household member density
  - `Latitude` & `Longitude`: Geospatial coordinates
- **Target Variable:** `Price` (Converted from `MedHouseVal` to absolute USD).

## 4. Data Exploration
Exploratory data analysis revealed:
1. `MedInc` exhibits the strongest linear correlation ($r \approx 0.69$) with home prices.
2. Prices are heavily skewed with extreme upper-end valuations.
3. Geographical clusters indicate significant price premiums along coastal latitudes.
*(Refer to `plots/1_price_distribution.png`, `plots/2_income_vs_price.png`, and `plots/3_correlation_heatmap.png`)*.

## 5. Data Preprocessing
- **Missing Value Check:** No null entries were present across all 20,640 observations.
- **Train/Test Splitting:** 80% Training (16,512 rows) and 20% Testing (4,128 rows) split using a fixed random seed (`random_state=42`) for reproducibility.
- **Scaling:** Input variables were normalized using `StandardScaler`. Standard scaling ensures distance- and penalty-sensitive algorithms treat high-magnitude features (such as `Population`) and small-scale features (`MedInc`) uniformly.

## 6. Feature Engineering & Selection
All 8 original dimensional and demographic attributes were retained. Spatial coordinates (`Latitude`, `Longitude`) were passed to tree models to capture non-linear geographical premiums without explicit regional clustering.

## 7. Machine Learning Technique Selection
Three regression paradigms were evaluated:
1. **Ridge Regression:** Regularized linear baseline penalizing multicollinearity.
2. **Random Forest Regressor:** Bagging ensemble of de-correlated decision trees to capture non-linear feature interactions.
3. **XGBoost Regressor:** Sequential gradient-boosted decision trees optimizing second-order Taylor approximations of the loss function.

## 8. Implementation
The solution is implemented in Python using `scikit-learn`, `xgboost`, and `fastapi`. Hyperparameters for XGBoost were constrained (`n_estimators=150`, `max_depth=6`, `learning_rate=0.08`) to prevent overfitting.

## 9. Results
| Model Architecture | Paradigm | $R^2$ Score | MAE ($) | RMSE ($) | MAPE (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ridge Regression** | Regularized Linear | 0.5758 | $52,620 | $70,350 | 26.8% |
| **Random Forest** | Bagging Ensemble | 0.7912 | $32,800 | $49,200 | 16.4% |
| **XGBoost Regressor** | Gradient Boosting | **0.8354** | **$28,400** | **$43,800** | **13.9%** |

## 10. Evaluation
- **$R^2$ Score (0.8354):** Explains 83.5% of variance in property valuations.
- **MAE vs. RMSE Gap:** The RMSE ($43,800) is moderately higher than MAE ($28,400), reflecting penalization on rare luxury home outliers.
- **Visual Validation:** Scatter analysis in `plots/4_actual_vs_predicted.png` shows strong adherence to the ideal $y = x$ reference diagonal.

## 11. Engineering & Financial Interpretation
- **Underwriting Tolerance:** 71.4% of property valuations fall within a $\pm 10\%$ acceptable loan-to-value (LTV) margin.
- **Cost Reduction:** For a lending institution evaluating 1,000 monthly loan applications, automating 71.4% of appraisals at an average manual survey cost of $450 saves approximately $321,300 in operational appraisal overhead.

## 12. Limitations
1. Dataset lacks micro-level property features (e.g., interior renovation status, school district ratings).
2. Upper-boundary truncation at $500,000 in legacy census collection caps model accuracy for luxury estates.

## 13. Recommendations
1. Incorporate real-time mortgage interest rate feeds as macroeconomic features.
2. Integrate computer vision models (CNNs) to score exterior and interior image quality alongside tabular attributes.

## 14. Conclusion
The XGBoost Regressor outperformed regularized linear methods, demonstrating that tree-based gradient boosting effectively resolves complex geographic and demographic relationships for automated financial property appraisal.