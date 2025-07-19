# Data Enrichment Agent ("Context Augmenter")

This agent enhances a cleaned dataset by adding external context, deriving new features, and merging it with related data to provide a richer foundation for analysis.

## Role in the Pipeline

The Context Augmenter receives a cleaned dataset from the Data Cleaning Agent ("Data Refinery"). It enriches this data before passing it to the Data Analysis Agent ("Trend Miner").

## Key Functions

*   **Merges with external data:** Joins the dataset with external sources (e.g., economic indicators, population data).
*   **Derives new features:** Calculates new columns like moving averages, year-over-year growth, or other relevant metrics.
*   **Annotates events:** Tags time periods with significant external events (e.g., market changes, policy updates).
*   **Fetches data from APIs:** Can be configured to call external APIs to fetch contextual information.

## Input

*   A path to a cleaned dataset file.

## Output

*   A path to the enriched dataset file.

## Dependencies

*   pandas
*   numpy
*   requests

## Configuration

*   **API Integration:** To fetch data from external APIs, you will need to configure the `tools.py` file with the necessary API endpoints and credentials.
    *   **TODO:** Add API keys and endpoint URLs to a secure configuration file or environment variables, and update `tools.py` to read from them.
*   **MCP Integration:** For more advanced use cases, such as querying a vector database for contextual news articles, an MCP server will need to be configured.
    *   **TODO:** Set up the MCP client in `agent.py` to connect to the appropriate MCP server and tool.
