import json
from pathlib import Path

class Load_config():
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self._load_config()

    def _load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, value in data.items():
            setattr(self, key, value)
    def save(self):
        data = {}
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                if isinstance(v, Path):
                    data[k] = str(v)
                else:
                    data[k] = v
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

