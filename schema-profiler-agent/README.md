# Schema Profiler Agent

This agent is responsible for intelligent dataset profiling using AI-powered analysis, with automatic configuration caching and reuse capabilities.

## Features

### 🤖 AI-Powered Profiling
*   **Google Gemini Integration**: Uses Gemini AI for sophisticated dataset analysis
*   **Intelligent Column Classification**: Automatically identifies dimensions, measures, dates, categories, IDs, and geographical data
*   **Business Context Analysis**: Provides business meaning and importance scoring for columns
*   **Relationship Detection**: Identifies hierarchies, correlations, and data relationships
*   **Analysis Recommendations**: Suggests specific analysis approaches based on data characteristics

### 💾 Configuration Caching & Reuse
*   **Smart Caching**: Automatically saves dataset configurations for future reuse
*   **Hash-based Validation**: Detects when dataset structure changes and regenerates configs
*   **Cross-Session Persistence**: Configurations persist across agent restarts
*   **Fallback Support**: Gracefully falls back to basic profiling if AI is unavailable

### 📊 Comprehensive Profiling
*   **Statistical Analysis**: Traditional column statistics and data quality assessment
*   **Schema Suggestions**: Primary key candidates, indexing recommendations
*   **Data Quality Scoring**: AI-enhanced quality assessment with issue detection
*   **Multiple Output Formats**: JSON configurations, YAML profiles, and structured metadata

## API Skills

### Core Profiling Skills
- **`ai_profile_dataset`**: AI-powered dataset profiling with configuration caching
- **`profile_dataset`**: Traditional statistical profiling
- **`get_column_statistics`**: Detailed analysis of specific columns
- **`compare_schemas`**: Compare schemas between datasets

### Configuration Management Skills
- **`get_dataset_config`**: Retrieve cached configuration for a dataset
- **`list_all_configs`**: List all cached dataset configurations

## Configuration Structure

Generated configurations include:

```json
{
  "dataset_name": "example_dataset",
  "dataset_hash": "abc123...",
  "configuration_date": "2024-01-15T10:30:00",
  "configuration_source": "gemini_intelligent_profiler",
  "column_mappings": {
    "primary_date": "order_date",
    "primary_metric": "revenue", 
    "all_metrics": ["revenue", "profit", "quantity"],
    "all_dimensions": ["country", "category", "customer_segment"],
    "geographical_dimensions": ["country", "state", "city"],
    "hierarchical_dimensions": ["category", "subcategory", "product"]
  },
  "ai_insights": {
    "dataset_type": "retail_sales",
    "domain": "e-commerce",
    "recommendations": [...]
  },
  "measure_metadata": {
    "revenue": {
      "type": "additive",
      "business_meaning": "Total sales revenue",
      "aggregation_method": "sum",
      "unit": "currency"
    }
  }
}
```

## Usage

### Environment Setup
1. Set `GOOGLE_API_KEY` environment variable or add to `.env` file
2. Install dependencies: `pip install google-generativeai`

### Basic Usage
```python
# AI-powered profiling with caching
result = await agent.ai_profile_dataset_skill(
    data_handle_id="dataset_123",
    use_cache=True,
    force_ai=False  # Falls back to basic if AI fails
)

# Check for existing configuration
config = await agent.get_dataset_config_skill(
    data_handle_id="dataset_123"
)
```

### Configuration Reuse
The agent automatically:
1. Checks for existing configurations before profiling
2. Validates dataset structure hasn't changed (via hash)
3. Reuses valid configurations for faster processing
4. Regenerates configurations when datasets change

This enables efficient reprocessing of known datasets and consistent analysis configurations across the multi-agent framework.
