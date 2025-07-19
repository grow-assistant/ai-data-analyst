import yaml
from pathlib import Path

def write_profile_yaml(output_dir: Path, dataset_name: str, statistical_profile: dict):
    output_path = output_dir / f"{dataset_name}.profile.yaml"
    with open(output_path, 'w') as f:
        yaml.dump(statistical_profile, f, default_flow_style=False)

def write_schema_yaml(output_dir: Path, dataset_name: str, semantic_profile: str):
    output_path = output_dir / f"{dataset_name}.schema.yaml"
    with open(output_path, 'w') as f:
        f.write(semantic_profile)

def write_yaml_files(output_dir: Path, dataset_name: str, statistical_profile: dict, semantic_profile: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    write_profile_yaml(output_dir, dataset_name, statistical_profile)
    write_schema_yaml(output_dir, dataset_name, semantic_profile)
