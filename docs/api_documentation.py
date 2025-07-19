"""
API Documentation Generator for MCP System.

This script automatically generates OpenAPI 3.0 specifications for all
agent and MCP server endpoints. It uses FastAPI's built-in OpenAPI
generation capabilities and combines them into a single, comprehensive
API documentation.

The generated documentation can be served via a simple web server
and provides an interactive interface (Swagger UI or Redoc) for
exploring and testing the APIs.
"""

import json
import os
import sys
from pathlib import Path
import importlib
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
import yaml

# Add project root to Python path to allow module imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common_utils.config_manager import initialize_config, get_config

def get_app_instance(module_path: str, app_variable: str = "app") -> FastAPI:
    """Dynamically import a FastAPI app instance from a module."""
    try:
        module = importlib.import_module(module_path)
        importlib.reload(module)  # Ensure we get the latest version
        app = getattr(module, app_variable)
        if not isinstance(app, FastAPI):
            raise TypeError(f"'{app_variable}' in {module_path} is not a FastAPI instance.")
        return app
    except (ImportError, AttributeError) as e:
        print(f"Error importing app from {module_path}: {e}")
        return None

def generate_openapi_spec(app: FastAPI, title: str, version: str, description: str) -> dict:
    """Generate OpenAPI specification for a FastAPI application."""
    if not app.openapi_schema:
        app.openapi_schema = get_openapi(
            title=title,
            version=version,
            description=description,
            routes=app.routes,
        )
    return app.openapi_schema

def merge_openapi_specs(base_spec: dict, specs_to_merge: dict) -> dict:
    """Merge multiple OpenAPI specs into a single spec."""
    merged_spec = base_spec.copy()
    
    # Merge paths
    if "paths" not in merged_spec:
        merged_spec["paths"] = {}
    for spec_name, spec in specs_to_merge.items():
        if "paths" in spec:
            for path, path_item in spec["paths"].items():
                # Add a tag to group endpoints by agent/server
                for method, operation in path_item.items():
                    if "tags" not in operation:
                        operation["tags"] = []
                    operation["tags"].insert(0, spec_name)
                
                # Add a prefix to the path to avoid collisions
                prefixed_path = f"/{spec_name}{path}"
                merged_spec["paths"][prefixed_path] = path_item

    # Merge components (schemas, security schemes, etc.)
    if "components" not in merged_spec:
        merged_spec["components"] = {}
    for spec_name, spec in specs_to_merge.items():
        if "components" in spec:
            for comp_type, comp_items in spec["components"].items():
                if comp_type not in merged_spec["components"]:
                    merged_spec["components"][comp_type] = {}
                for comp_name, comp_def in comp_items.items():
                    # Prefix component names to avoid collisions
                    prefixed_comp_name = f"{spec_name.replace('_', ' ').title().replace(' ', '')}{comp_name}"
                    merged_spec["components"][comp_type][prefixed_comp_name] = comp_def
    
    # Update references in the merged spec
    merged_spec_str = json.dumps(merged_spec)
    for spec_name, spec in specs_to_merge.items():
         if "components" in spec:
            for comp_type, comp_items in spec["components"].items():
                for comp_name in comp_items.keys():
                    original_ref = f"#/components/{comp_type}/{comp_name}"
                    prefixed_comp_name = f"{spec_name.replace('_', ' ').title().replace(' ', '')}{comp_name}"
                    new_ref = f"#/components/{comp_type}/{prefixed_comp_name}"
                    merged_spec_str = merged_spec_str.replace(f'"{original_ref}"', f'"{new_ref}"')

    return json.loads(merged_spec_str)


def main():
    """Main function to generate API documentation."""
    print("Starting API documentation generation...")
    
    # Initialize configuration
    config_file = project_root / "config" / "system_config.yaml"
    if not config_file.exists():
        print(f"Error: Configuration file not found at {config_file}")
        return
        
    try:
        initialize_config(config_file)
    except Exception as e:
        print(f"Error initializing configuration: {e}")
        return

    # Base OpenAPI specification
    system_config = get_config("system")
    base_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": f"{system_config.get('name', 'MCP System')} - Combined API",
            "version": system_config.get('version', '1.0.0'),
            "description": "This documentation combines APIs for the MCP Server and all related agents.",
        },
        "tags": [],
    }
    
    specs_to_merge = {}
    
    # Generate spec for MCP Server
    mcp_config = get_config("mcp_server")
    # This assumes the server is defined in a standard way.
    # We might need to adjust module paths if they are different.
    mcp_app_module = "data_loader.mcp_server.server"
    mcp_app = get_app_instance(mcp_app_module)
    if mcp_app:
        mcp_spec = generate_openapi_spec(
            app=mcp_app,
            title="MCP Server API",
            version=system_config.get('version', '1.0.0'),
            description="API for the Multi-Agent Control Plane server."
        )
        specs_to_merge["mcp_server"] = mcp_spec
        base_spec["tags"].append({"name": "mcp_server", "description": "Endpoints for the MCP Server"})

    # Generate specs for all agents
    agents_config = get_config("agents", {})
    for agent_name, agent_config in agents_config.items():
        agent_app_module = agent_config.get("module_path")
        if not agent_app_module:
            print(f"Skipping agent '{agent_name}': no module_path configured.")
            continue
            
        # The main entry point for agents is often __main__.py
        # We assume the FastAPI app is defined in a file that can be imported.
        # Let's try to find the app in the agent's main module.
        try:
            agent_app = get_app_instance(f"{agent_app_module}.__main__")
        except (ImportError, TypeError):
             try:
                agent_app = get_app_instance(f"{agent_app_module}.agent")
             except (ImportError, TypeError):
                print(f"Could not find FastAPI app for agent '{agent_name}'")
                continue

        if agent_app:
            agent_spec = generate_openapi_spec(
                app=agent_app,
                title=f"{agent_config.get('name', agent_name)} API",
                version=system_config.get('version', '1.0.0'),
                description=f"API for the {agent_name}."
            )
            specs_to_merge[agent_name] = agent_spec
            base_spec["tags"].append({"name": agent_name, "description": f"Endpoints for the {agent_name}"})

    # Merge all specs
    print(f"Found {len(specs_to_merge)} individual OpenAPI specs to merge.")
    final_spec = merge_openapi_specs(base_spec, specs_to_merge)
    
    # Save the final spec
    output_dir = project_root / "docs" / "api"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_output_path = output_dir / "openapi.json"
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(final_spec, f, indent=2)
    print(f"✅ Combined OpenAPI JSON saved to: {json_output_path}")

    # Save as YAML
    yaml_output_path = output_dir / "openapi.yaml"
    with open(yaml_output_path, "w", encoding="utf-8") as f:
        yaml.dump(final_spec, f, allow_unicode=True, sort_keys=False)
    print(f"✅ Combined OpenAPI YAML saved to: {yaml_output_path}")

    # Generate a simple HTML page to serve Swagger UI
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <title>{base_spec['info']['title']}</title>
      <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@3/swagger-ui.css">
    </head>
    <body>
      <div id="swagger-ui"></div>
      <script src="https://unpkg.com/swagger-ui-dist@3/swagger-ui-bundle.js"></script>
      <script>
        const ui = SwaggerUIBundle({{
          url: "./openapi.json",
          dom_id: '#swagger-ui',
          presets: [
            SwaggerUIBundle.presets.apis,
            SwaggerUIBundle.SwaggerUIStandalonePreset
          ],
          layout: "StandaloneLayout"
        }})
      </script>
    </body>
    </html>
    """
    html_output_path = output_dir / "index.html"
    with open(html_output_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"✅ Swagger UI HTML page saved to: {html_output_path}")
    
    print("\nTo view the documentation, open 'docs/api/index.html' in your browser.")
    print("You may need to run a local web server, e.g., 'python -m http.server' from the 'docs/api' directory.")


if __name__ == "__main__":
    main() 