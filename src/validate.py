import os
from ultralytics import YOLO

def validate_model(model_path, yaml_path):
    '''
    This will validate the model at model_path against the currently generated attack/vandalized images
    dataset.
    '''
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return
    
    if not os.path.exists(yaml_path):
        print(f"❌ Error: YAML not found at {yaml_path}")
        return

    # load model
    model = YOLO(model_path)

    # evaluate
    model.val(data=yaml_path)

'''
# TODO: MAKE THIS SO IT AUTO AUTORUNS COMPARISONS OF EACH MODEL AND PROVIDES CLEAN SUMMARY.
Ex: 
1. Runs baseline on clean dataset, tape dataset, adversarial patch dataset, etc.
2. Runs 50_50_split_tape on clean dataset, tape dataset, adversarial patch dataset, etc. 
3. Compares the maP50 scores of all models for each dataset in a nice summary

'''
if __name__ == "__main__":
    # get project root
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # DEFINE PATHS, IF ADDING A NEW TRAINED MODEL DEFINE ITS PATH HERE UNDER 'MODEL_NAME' = .......:
    BASELINE_MODEL = os.path.join(ROOT, 'models', 'best.pt')
    FIFTY_FIFTY_TAPE_MODEL = os.path.join(ROOT, 'models', 'fifty_fifty_split_tape.pt')

    # Datasets
    ATTACK_YAML = os.path.join(ROOT, 'datasets', 'archive', 'attack.yaml')
    DATA_YAML = os.path.join(ROOT, 'datasets', 'archive', 'data.yaml')

    # Run it
    validate_model(FIFTY_FIFTY_TAPE_MODEL, DATA_YAML)