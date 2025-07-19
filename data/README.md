# 📊 Raw Datasets Directory

This directory is intended to store the large, raw datasets that you want to analyze with the Multi-Agent Data Analysis Framework.

## ❗ Important Note

**This directory is excluded from version control by the `.gitignore` file.**

Storing large data files directly in a Git repository is not recommended because it can make the repository very large and slow to clone.

## 📋 How to Use

1.  **Place your datasets here**:
    *   `Superstore.csv`
    *   `M5.csv`
    *   `AI_DS.tdsx`
    *   Any other `.csv`, `.json`, or `.tdsx` files you want to analyze.

2.  **Run the analysis**:
    *   The Streamlit dashboard and the framework CLI will automatically look for datasets in this directory.

## 📂 Example Datasets

The framework is tested with the following datasets:

*   **`Superstore.csv`**: A standard sample dataset for sales analysis.
*   **`M5.csv`**: A large time-series dataset for forecasting.
*   **`AI_DS.tdsx`**: A Tableau data source file for testing Hyper API integration.

## ☁️ Data Storage Recommendations

For collaborative projects, it is recommended to store large datasets in a shared location, such as:

*   **Cloud Storage**: Google Cloud Storage, Amazon S3, Azure Blob Storage
*   **Data Warehouse**: BigQuery, Snowflake, Redshift
*   **Shared Network Drive**

You can then write a simple script to download the required datasets into this directory before running the analysis.

---
*This file is here to explain the purpose of the `data/` directory. It is safe to delete if you no longer need this explanation.* 