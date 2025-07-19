# Personal Assistant

This sample demonstrates the use of the Agent Development Kit to create a personal assistant. The personal assistant can help you with your daily tasks by using its sub-agents to perform various actions.

## Overview

This personal assistant has two sub-agents:

*   **Google Search:** This sub-agent can search Google for information.
*   **Email Reviewer:** This sub-agent can review your emails for important information.

## Setup and Installation

### Prerequisites

*   Python 3.11+
*   Google Cloud Project (for Vertex AI integration)
*   API Key for Google Cloud
*   Google Agent Development Kit 1.0+
*   Poetry: Install Poetry by following the instructions on the official Poetry [website](https://python-poetry.org/docs/). For Windows PowerShell, you can run: `(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -`. You may need to restart your terminal after installation.

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/google/adk-samples.git
    cd adk-samples/python/agents/base-multi-agent
    ```
    NOTE: From here on, all command-line instructions shall be executed under the directory `base-multi-agent/` unless otherwise stated.

2.  Install dependencies using Poetry:

    ```bash
    poetry install
    ```

3.  Set up your environment variables:

    - At the top directory `base-multi-agent/`, make a `.env` by copying `env.example`
    - Set the following environment variables:
    ```
    # Google Search
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

    # Email credentials
    EMAIL_ADDRESS="YOUR_EMAIL_ADDRESS"
    EMAIL_PASSWORD="YOUR_EMAIL_PASSWORD"
    ```

4.  Activate the virtual environment set up by Poetry, run:
    ```bash
    poetry shell
    ```
    This command will spawn a new shell with the project's virtual environment activated. You will know it's active because your command prompt will likely change. Repeat this command whenever you have a new shell, before running the commands in this README.

## Running the Agent

### Using `adk`

ADK provides convenient ways to bring up agents locally and interact with them.
You may talk to the agent using the CLI:

```bash
# Under the base-multi-agent directory:
adk run personal_assistant
```

or via its web interface:
```bash
# Under the base-multi-agent directory:
adk web
```

This will start a local web server on your machine. You may open the URL, select "personal_assistant" in the top-left drop-down menu, and
a chatbot interface will appear on the right.