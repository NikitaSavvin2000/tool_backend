from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # Корень проекта
CONFIG_DIR = PROJECT_ROOT / "src" / "configuration"

def load_possible_cols():
    config_path = CONFIG_DIR / "possible_cols.yaml"
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config.get('all_possible_cols', [])
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
