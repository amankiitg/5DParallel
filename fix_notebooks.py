import nbformat
from pathlib import Path

def clean_notebook(path):
    nb = nbformat.read(path, as_version=nbformat.NO_CONVERT)

    # If widgets metadata exists but is missing "state", remove it
    if "widgets" in nb["metadata"]:
        widgets = nb["metadata"]["widgets"]
        if isinstance(widgets, dict) and "state" not in widgets:
            print(f"Fixing widgets metadata in {path}")
            del nb["metadata"]["widgets"]

    nbformat.write(nb, path)

if __name__ == "__main__":
    repo_path = Path(".")  # repo root
    notebooks = list(repo_path.rglob("*.ipynb"))

    for nb_file in notebooks:
        try:
            clean_notebook(nb_file)
        except Exception as e:
            print(f"Error fixing {nb_file}: {e}")
