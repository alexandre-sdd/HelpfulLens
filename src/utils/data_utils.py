from pathlib import Path


# ---------- Paths helpers ----------

def get_project_root() -> Path:
    """
    Infer project root by walking up until we find a 'data' folder.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "data").exists():
            return parent
    raise RuntimeError("Could not locate project root (no 'data/' folder found).")