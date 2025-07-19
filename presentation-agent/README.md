# Presentation Agent ("Insight Reporter")

This agent is responsible for taking the final analysis results and presenting them in a human-friendly format. It can generate reports, create visualizations, and craft a narrative to explain the findings.

## Role in the Pipeline

The Insight Reporter is the final agent in the workflow. It receives structured analytical data from the Data Analysis Agent ("Trend Miner") and produces the final output for the end-user.

## Key Functions

*   **Generates visualizations:** Creates charts and graphs (e.g., line charts, bar charts) to highlight key trends.
*   **Crafts narratives:** Uses a large language model (LLM) to generate explanatory text and summaries of the findings.
*   **Compiles reports:** Assembles the visualizations and narrative into a single report, which can be in various formats like HTML or Markdown.

## Input

*   A JSON object containing the structured analysis results from the Trend Miner. This should include KPIs, identified trends, anomalies, and any other key findings.

## Output

*   A final report file (e.g., `report.html` or `report.md`).

## Dependencies

*   matplotlib
*   plotly
*   Jinja2
*   openai

## Configuration

*   **LLM API Key:** This agent requires an API key for a large language model (e.g., OpenAI) to generate narratives.
    *   **TODO:** Add your LLM API key to a secure configuration file or environment variable and update `tools.py` to use it.
