# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Orchestrator agent using MCP for inter-agent communication"""

import logging
from google.adk.agents import Agent
from .prompt import ROOT_AGENT_INSTR
from .a2a_client import OrchestratorA2AClient
from common_utils.types import TaskRequest, TaskResponse
from typing import Dict, Optional
import uuid
import asyncio

# Import configuration from common utils
from common_utils.config import settings

logger = logging.getLogger(__name__)
logger.info("Orchestrator agent using MCP-only communication")

async def route_task_via_a2a(task_type: str, description: str, data_handles: Optional[list] = None) -> str:
    """
    Route a task to an appropriate agent via A2A protocol.
    
    Args:
        task_type: Type of task (e.g., "loading", "analysis", "general")
        description: Task description
        data_handles: Optional list of data handles to pass with the task
        
    Returns:
        Response content from the target agent via A2A
    """
    try:
        logger.info(f"Routing {task_type} task via A2A: {description[:50]}...")
        
        # Map task_type to agent and skill
        agent_map = {
            "loading": ("data_loader_agent", "load_dataset"),
            "cleaning": ("data_cleaning_agent", "clean_dataset"),
            "enrichment": ("data_enrichment_agent", "enrich_dataset"),
            "analysis": ("data_analyst_agent", "analyze_dataset"),
            "presentation": ("presentation_agent", "create_report"),
            "general": ("orchestrator_agent", "general_task")  # Fallback
        }
        
        if task_type not in agent_map:
            return f"Error: Unknown task type {task_type}"
        
        agent_name, skill_id = agent_map[task_type]
        
        parameters = {
            "description": description,
            "data_handles": data_handles or []
        }
        
        a2a_client = OrchestratorA2AClient()
        result = await a2a_client.send_task_to_agent(agent_name, skill_id, parameters)
        
        if "error" in result:
            return f"Error routing task via A2A: {result['error']}"
        
        logger.info(f"Successfully received response via A2A from {agent_name}")
        return str(result)
        
    except Exception as e:
        logger.error(f"Failed to route task via A2A: {e}")
        return f"Error: Failed to route task via A2A - {str(e)}"

def create_orchestrator_agent():
    """Create the orchestrator agent with MCP-based tools."""
    return Agent(
        model=settings.agent_model,
        name=settings.agent_name,
        description="Orchestrates tasks between agents using the Multi-Agent Control Plane (MCP).",
        instruction=ROOT_AGENT_INSTR.replace("MCP", "A2A"),
        tools=[route_task_via_a2a],
        sub_agents=[]  # No sub-agents
    )

# Create the agent instance
root_agent = create_orchestrator_agent()
