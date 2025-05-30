import os

# Define the folder and file structure
structure = [
    "main.py",
    "requirements.txt",
    {"models": [
        "object_detection.py",
        "scene_description.py",
        "ocr.py"
    ]},
    {"schemas": [
        "schemas.py"
    ]},
    {"utils": [
        "image_processing.py",
        "text_generation.py",
        "settings_manager.py"
    ]},
    {"data": [
        "user_settings.json",
        "hazard_list.json"
    ]}
]

def create_structure(base_path, tree):
    for item in tree:
        if isinstance(item, str):
            file_path = os.path.join(base_path, item)
            open(file_path, 'a').close()
        elif isinstance(item, dict):
            for folder, contents in item.items():
                folder_path = os.path.join(base_path, folder)
                os.makedirs(folder_path, exist_ok=True)
                create_structure(folder_path, contents)

# Run the creation script
base_dir = os.getcwd()  # or set a specific path
project_root = os.path.join(base_dir)
os.makedirs(project_root, exist_ok=True)

create_structure(project_root, structure)

print("✅ Project structure created successfully.")
