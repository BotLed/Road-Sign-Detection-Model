import os
from ultralytics import YOLO

def validate_model(model_name, model_path, dataset_name, yaml_path):
    '''
    Validates the model and returns dictionary of key metrics.
    '''
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return None
    
    if not os.path.exists(yaml_path):
        print(f"❌ Error: YAML not found at {yaml_path}")
        return None

    # load model
    model = YOLO(model_path)

    # verbose=False to keep console clean for the summary
    results = model.val(data=yaml_path, verbose=False)

    return {
        "Model": model_name,
        "Dataset": dataset_name,
        "mAP50": results.results_dict['metrics/mAP50(B)'],
        "mAP50-95": results.results_dict['metrics/mAP50-95(B)'],
        "Precision": results.results_dict['metrics/precision(B)'],
        "Recall": results.results_dict['metrics/recall(B)']
    }


def print_summary(all_results):
    print("\n" + "="*85)
    print(f"{'MODEL':<25} | {'DATASET':<15} | {'mAP50':<10} | {'mAP50-95':<10} | {'Prec':<10}")
    print("-" * 85)
    for res in all_results:
        if res:
            print(f"{res['Model']:<25} | {res['Dataset']:<15} | {res['mAP50']:<10.4f} | {res['mAP50-95']:<10.4f} | {res['Precision']:<10.4f}")
    print("="*85 + "\n")


if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # model paths -> NOTE: IF ADDING A NEW MODEL ADD IT HERE
    models = {
    "Baseline": os.path.join(ROOT, 'models', 'best.pt'),
    "50/50 Tape": os.path.join(ROOT, 'models', 'fifty_fifty_split_tape.pt'),
    "70/30 Tape": os.path.join(ROOT, 'models', '70_30_tape.pt'),
    "50/50 Patch": os.path.join(ROOT, 'models', '50_50_patch.pt'),
    "70/30 Patch": os.path.join(ROOT, 'models', '70_30_patch.pt'),
    "50/50 Illumination": os.path.join(ROOT, 'models', '50_50_illumination.pt'),
    "70/30 Illumination": os.path.join(ROOT, 'models', '70_30_illumination.pt'),
    "70/30 Graffiti": os.path.join(ROOT, 'models', '70_30_graffiti.pt'),
    "50/50 Mixed":    os.path.join(ROOT, 'models', '50_50_mixed.pt'),
    }

    # dataset paths, this doesn't really have to be changed unless we change the adversarial attack dataset creation logic to not overwrite
    datasets = {
        "Clean": os.path.join(ROOT, 'datasets', 'archive', 'data.yaml'),
        "Attack": os.path.join(ROOT, 'datasets', 'archive', 'attack.yaml')
    }

    results_summary = []

    for m_name, m_path in models.items():
        for d_name, d_path in datasets.items():
            print(f"Evaluating {m_name} on {d_name} dataset...")
            res = validate_model(m_name, m_path, d_name, d_path)
            results_summary.append(res)

    print_summary(results_summary)