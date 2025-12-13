import kagglehub
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOCAL_FOLDER_NAME = "dataset"
MAIN_DATA_FOLDER = "COVID-19_Radiography_Dataset"
EXACT_FILES_TO_REMOVE = [
    "COVID.metadata.xlsx",
    "README.md.txt",
    "Lung_Opacity.metadata.xlsx",
    "Normal.metadata.xlsx",
    "Viral Pneumonia.metadata.xlsx",
]
FOLDER_TO_REMOVE = "Lung_Opacity"

# [CHECK IF DATASET ALREADY EXIST]
destination_path = os.path.join(os.getcwd(), LOCAL_FOLDER_NAME)
if os.path.isdir(destination_path) and len(os.listdir(destination_path)) > 0:
    print(
        f"Local dataset folder already exists and is not empty at: {destination_path}"
    )
    sys.exit()

# [MOVE DATASET FORM DOWNLOAD FOLDER TO WORKING DIRECTORY]
os.makedirs(destination_path, exist_ok=True)
cache_path = kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")
source_data_folder = os.path.join(cache_path, MAIN_DATA_FOLDER)
move_source = source_data_folder
for item_name in os.listdir(move_source):
    source = os.path.join(move_source, item_name)
    destination = os.path.join(destination_path, item_name)
    shutil.move(source, destination)
print(f"All contents successfully moved to: {destination_path}")

# [CLEAN UP THE DATASET]
os.rename(
    os.path.join(destination_path, "Viral Pneumonia"),
    os.path.join(destination_path, "Non-COVID"),
)
os.rename(
    os.path.join(destination_path, "Normal"), os.path.join(destination_path, "Healthy")
)
shutil.rmtree(os.path.join(destination_path, FOLDER_TO_REMOVE))
for file_name in EXACT_FILES_TO_REMOVE:
    file_path = os.path.join(destination_path, file_name)
    os.remove(file_path)
shutil.rmtree(move_source)
shutil.rmtree(cache_path)
print("DONE!")
