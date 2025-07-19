SCHEMA_PROFILING_PROMPT = """
You are an expert data analyst. Your task is to analyze the schema and profile of a new dataset and infer its semantic structure.

Based on the column names, data types, and top values provided below, identify the following:
1.  **Primary Key**: The column that most likely serves as the unique identifier for each row.
2.  **Dimensions**: Categorical columns used for slicing and dicing data (e.g., Country, Category).
3.  **Measures**: Numerical columns that can be aggregated (e.g., Sales, Quantity).
4.  **Hierarchies**: Logical drill-down paths (e.g., Year -> Quarter -> Month).
5.  **Semantic Types**: The business meaning of a column (e.g., geo, date, currency).

**Input Data Profile:**
```json
{profile}
```

**Instructions:**
- Analyze the input profile carefully.
- For each column, determine its most likely role (dimension or measure).
- Identify potential hierarchies between dimension columns.
- Assign a semantic type to columns where applicable.
- Return your analysis in a structured YAML format, like the example below.

**Example Output:**
```yaml
dataset: sales_orders
primary_key: order_id
dimensions:
  - name: order_date
    semantic_type: date
    hierarchy: [year, quarter, month, day]
  - name: country
    semantic_type: geo_iso3
  - name: product_category
    semantic_type: category
measures:
  - name: revenue
    data_type: decimal
    aggregation: sum
  - name: quantity
    data_type: int
    aggregation: sum
```

**Your Analysis (in YAML format):**
"""
