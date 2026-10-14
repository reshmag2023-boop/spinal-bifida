"""
Utility script to create a clean, complete ZIP archive of the Spina Bifida system.
Excludes .git, .venv, __pycache__, and temporary caches.
"""

import os
import zipfile
from pathlib import Path

def create_project_zip():
    base_dir = Path(r"C:\Users\reshm\Downloads\SpinaBifida_System")
    zip_output_dir = Path(r"C:\Users\reshm\Downloads\AISKILLPULSE")
    zip_path = zip_output_dir / "SpinaBifida_Explainable_AI_System.zip"

    exclude_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", ".idea", ".vscode"}
    exclude_exts = {".pyc", ".log"}

    print(f"Creating ZIP archive at: {zip_path}")
    count = 0
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dir):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if any(file.endswith(ext) for ext in exclude_exts):
                    continue
                file_path = Path(root) / file
                # Relative path inside zip under SpinaBifida_System/
                rel_path = file_path.relative_to(base_dir)
                archive_name = Path("SpinaBifida_System") / rel_path
                zipf.write(file_path, str(archive_name))
                count += 1
                if count % 500 == 0:
                    print(f"Archived {count} files...")

    # Also copy to SpinaBifida_System root
    system_zip_path = base_dir / "SpinaBifida_Explainable_AI_System.zip"
    import shutil
    shutil.copy2(str(zip_path), str(system_zip_path))

    zip_size_mb = os.path.getsize(str(zip_path)) / (1024 * 1024)
    print(f"\n[OK] Successfully packaged {count} files into:")
    print(f"  1. {zip_path} ({zip_size_mb:.1f} MB)")
    print(f"  2. {system_zip_path} ({zip_size_mb:.1f} MB)")

if __name__ == "__main__":
    create_project_zip()
