import os, json
from pathlib import Path

import numpy as np
import pandas as pd

# helpers
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)