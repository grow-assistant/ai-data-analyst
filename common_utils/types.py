# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common data schema and types for data analysis agents."""

from typing import Optional, Union, Any, Dict, List
from pydantic import BaseModel, Field


class DataFileInfo(BaseModel):
    """Information about a data file."""
    filename: str = Field(description="Name of the data file")
    file_type: str = Field(description="Type of file (csv, json, xlsx, etc.)")
    size_bytes: int = Field(description="File size in bytes")
    last_modified: Optional[str] = Field(description="Last modification timestamp")


class DataSchema(BaseModel):
    """Schema information for a dataset."""
    columns: List[str] = Field(description="List of column names")
    data_types: Dict[str, str] = Field(description="Data types for each column")
    row_count: int = Field(description="Number of rows in the dataset")
    missing_values: Dict[str, int] = Field(description="Count of missing values per column")


class AnalysisResult(BaseModel):
    """Result of a data analysis operation."""
    analysis_type: str = Field(description="Type of analysis performed")
    results: Dict[str, Any] = Field(description="Analysis results")
    visualization_data: Optional[Dict[str, Any]] = Field(description="Data for visualizations")
    summary: str = Field(description="Human-readable summary of results")


class DataMetadata(BaseModel):
    """Metadata for a dataset."""
    name: str
    description: str
    source: str
    schema: Dict[str, Any]
    row_count: int
    column_count: int
    data_types: Dict[str, str]
    
    
class DataHandle(BaseModel):
    """Handle for efficiently passing data between agents."""
    handle_id: str = Field(description="Unique handle identifier")
    data_type: str = Field(description="Type of data (csv, json, dataframe, etc.)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Data metadata")
    created_at: str = Field(description="Creation timestamp")
    expires_at: Optional[str] = Field(description="Expiration timestamp")
    size_bytes: int = Field(description="Data size in bytes")


# Agent-related types (moved from deprecated A2A module)
class AgentSkill(BaseModel):
    """Agent skill definition."""
    id: str
    name: str
    description: str
    tags: List[str]


class AgentCapabilities(BaseModel):
    """Agent capabilities definition."""
    pass


class AgentCard(BaseModel):
    """Agent card for discovery and routing."""
    name: str
    description: str
    url: str
    version: str
    capabilities: AgentCapabilities
    defaultInputModes: List[str]
    defaultOutputModes: List[str]
    skills: List[AgentSkill]


class TaskRequest(BaseModel):
    """Request for task execution between agents."""
    task_id: str = Field(description="Unique task identifier")
    trace_id: str = Field(description="Unique identifier for tracing a request across multiple agents")
    task_type: str = Field(description="Type of task to execute")
    parameters: Dict[str, Any] = Field(description="Task parameters")
    data_handles: List[DataHandle] = Field(default_factory=list, description="Data references")
    priority: int = Field(default=5, description="Task priority (1-10)")


class TaskResponse(BaseModel):
    """Response from task execution."""
    task_id: str = Field(description="Original task identifier")
    trace_id: str = Field(description="Identifier for tracing a request across multiple agents")
    status: str = Field(description="Task status (success, error, pending)")
    agent_name: str = Field(description="Name of the agent that executed the task")
    results: Optional[Dict[str, Any]] = Field(description="Task results")
    error_message: Optional[str] = Field(description="Error message if failed")
    data_handles: List[DataHandle] = Field(default_factory=list, description="Result data references")
    execution_time_ms: Optional[int] = Field(description="Execution time in milliseconds") 