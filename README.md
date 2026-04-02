# Retail Demand Prediction Study

---

## 1. Retail Demand Prediction Using Random Forest Regressor

A machine learning project that predicts retail demand using Random Forest Regressor. This study target to train a model on historical sales data to predict future product demand, helping retailers optimize inventory management and reduce manual ordering time.

**Overview:**

Retail companies often struggle with inventory management due to demand fluctuations. This study addresses the problem by building a machine learning model that predicts regular demand based on past sales patterns. The model outperforms traditional system calculations by accounting for demand patterns influenced by stockouts.

**Dataset:**

Simulated retail store sales data for a specific product category

```
# Folder sturcture to run the notebook locally
myproject/
├── demand_prediction.ipynb
├── prediction.py
├── dataset/
│   └── sales_data.csv
├── evaluate/
│   ├── evaluation_data.csv
├── input/
│   └── new_sales_data.csv
├── model/ 
└── output/
```

**Folder Description:**

- **demand_prediction.ipynb**: Main Jupyter Notebook containing the complete analysis, preprocessing, model training, and evaluation
- **prediction.py**: Python script for making predictions on new sales data using the trained model
- **dataset/**: Contains the training dataset (sales_data.csv)
- **evaluate/**: Contains evaluation data and evaluation prediction results
- **input/**: Directory for new sales input data (eg. weekly CSV files from database)
- **model/**: Stores the trained model
- **output/**: Output directory for prediction results

**Required Packages:**
```
numpy pandas scikit-learn matplotlib seaborn joblib
```
**Project Workflow:**

1. Data Preprocessing
2. Model Training
3. Model Monitoring
4. Prediction Pipeline
5. Model Evaluation

---
**Result:**

Accuracy Achievement: the % of no. of estimations within ±20% vs actual demand (higher % is better):
| Method | within ±20% |
|---------|---------|
| Model | 96.19% |
| System | 62.86% |

**Key Insights:**

1. Better Inventory Control: Reduced overstock and stockout situations
2. Improved Service Level: More accurate demand forecasts
3. Time Savings: Automates manual ordering process, freeing staff for higher-value tasks
4. Data-Driven Decisions: Based on historical patterns rather than experience

**Suggestion for Future Enhancements:**

- Incorporate seasonality detection and handling
- Add external factors (marketing campaigns, holidays)
- Implement automated retraining pipeline

---