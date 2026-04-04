import os
from ultralytics import YOLO

def validate_model(model_path, yaml_path):
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return
    
    if not os.path.exists(yaml_path):
        print(f"❌ Error: YAML not found at {yaml_path}")
        return

    # load model (rn only best.pt is available)
    model = YOLO(model_path)

    # evaluate
    model.val(data=yaml_path)

if __name__ == "__main__":
    # get project root
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # DEFINE PATHS, IF ADDING A NEW TRAINED MODEL DEFINE ITS PATH HERE UNDER 'MODEL_NAME' = .......:
    BASELINE_MODEL = os.path.join(ROOT, 'models', 'best.pt')
    ATTACK_YAML = os.path.join(ROOT, 'datasets', 'archive', 'attack.yaml')

    # Run it
    validate_model(BASELINE_MODEL, ATTACK_YAML)