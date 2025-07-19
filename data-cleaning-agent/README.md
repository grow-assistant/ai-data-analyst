# Data Cleaning Agent ("Data Refinery")

This agent is responsible for cleaning and preprocessing raw data. It takes a dataset as input, performs various cleaning operations, and outputs a cleaned, analysis-ready dataset.

## Role in the Pipeline

The Data Refinery receives raw data from the Data Ingestion Agent ("Archive Harvester"). Its primary function is to ensure data quality and consistency before the data is passed to the Data Enrichment Agent.

## Key Functions

*   Handles missing values (e.g., imputation with mean, median, or a constant).
*   Removes or smooths outliers.
*   Corrects inconsistent data entries and standardizes formats (e.g., dates, units).
*   Filters out irrelevant records or columns.

## Input

*   A path to a raw dataset file (e.g., CSV, Parquet).

## Output

*   A path to the cleaned dataset file.
*   A JSON summary of the cleaning operations performed.

## Dependencies

*   pandas
*   numpy
