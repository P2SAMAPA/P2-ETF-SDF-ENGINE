# configs.py
import yaml
import os

# Load configuration from YAML file
_config_path = os.path.join(os.path.dirname(__file__), 'configs.yaml')
with open(_config_path, 'r') as f:
    CONFIG = yaml.safe_load(f)
