import os
import shutil

# Directories to delete
dirs_to_delete = ["build", "install", "log"]

for dir_name in dirs_to_delete:
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
        print(f"Deleted: {dir_name}")
    else:
        print(f"Not found: {dir_name}")
