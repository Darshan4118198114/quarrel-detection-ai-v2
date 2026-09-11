"""
Builds a clean, shareable ZIP package of the Quarrel Detection AI project.
Excludes virtual environments (venv, qd), git repositories, caches, and IDE metadata.
"""

import os
import zipfile

OUTPUT_ZIP = "quarrel_detection_ai.zip"

EXCLUDE_DIRS = {
    "venv",
    "qd",
    ".git",
    "__pycache__",
    ".vscode",
    "test_incidents"
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd"
}

EXCLUDE_FILES = {
    OUTPUT_ZIP,
    ".gitignore"
}

def make_clean_zip():
    root_dir = os.path.abspath(os.path.dirname(__file__))
    zip_path = os.path.join(root_dir, OUTPUT_ZIP)

    print(f"Creating clean package: {OUTPUT_ZIP} ...")
    file_count = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for current_root, dirs, files in os.walk(root_dir):
            # Prune excluded directories in-place so os.walk does not descend into them
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith("__pycache__")]

            for f in files:
                if f in EXCLUDE_FILES or f == OUTPUT_ZIP:
                    continue
                ext = os.path.splitext(f)[1].lower()
                if ext in EXCLUDE_EXTENSIONS:
                    continue

                full_path = os.path.join(current_root, f)
                rel_path = os.path.relpath(full_path, root_dir)

                # Store with project root folder name for tidy extraction
                archive_name = os.path.join("quarrel-detection-ai", rel_path)
                zf.write(full_path, archive_name)
                print(f"  + Added: {archive_name}")
                file_count += 1

    zip_size_kb = os.path.getsize(zip_path) / 1024.0
    print(f"\n[SUCCESS] Package successfully created!")
    print(f"  Filename: {zip_path}")
    print(f"  Files packaged: {file_count}")
    print(f"  Total Size: {zip_size_kb:.1f} KB")

if __name__ == "__main__":
    make_clean_zip()
