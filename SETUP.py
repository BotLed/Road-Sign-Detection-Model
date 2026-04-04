import zipfile
import os
from  src.attack import generate_attack_dataset, create_attack_yaml

'''
BEFORE RUNNING THIS MAKE SURE TO 
1. DOWNLOAD ARCHIVE.ZIP FROM: https://www.kaggle.com/datasets/lara311/traffic-sign-recognition-yolov8
2. PASTE THE UNZIPPED FOLDER IN THE TLD OF THIS REPOSITORY

AFTER RUNNING:
1. TAKE THE PROVIDED DATA.YAML IN THE TLD AND PASTE IT INTO datasets/archive, REPLACING
   THE OLD DATA.YAML. THIS HAS TO BE DONE.

TO VALIDATE: RUN VALIDATE.PY
'''

def setup_workspace():
    # 1. gets directory where THIS script is saved
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # defines paths relative to base_dir
    zip_path = os.path.join(base_dir, 'archive.zip')
    extract_path = os.path.join(base_dir, 'datasets', 'archive')

    # check if archive.zip exists, YOU NEED IT IN THE SAME LEVEL AS THIS SCRIPT (unzipped)
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path} to {extract_path}...")
        
        # ensure destination folder exists
        os.makedirs(extract_path, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
        print("✅ Extraction complete")

        '''
        CHANGE "patch" TO WHICHEVER ATTACK TYPE YOU WOULD LIKE TO VALIDATE ON (possible types found in src/attack.py)
        '''
        generate_attack_dataset(extract_path, "tape")
        create_attack_yaml(extract_path)

    else:
        print(f"❌ Error: Could not find '{zip_path}' in the current directory.")

if __name__ == '__main__':
    setup_workspace()