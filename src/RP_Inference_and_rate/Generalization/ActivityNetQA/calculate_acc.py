import json
import re

def calculate_accuracy(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    correct = 0
    total = len(data)
    failed_extractions = 0

    print(f"Start validations: {json_file}")
    print("-" * 30)

    for entry in data:
        gt_answer = str(entry.get("answer", "")).strip().lower()
        
        raw_output = entry.get("model_raw_output", "")
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', raw_output, re.IGNORECASE | re.DOTALL)
        
        if match:
            pred_answer = match.group(1).strip().lower()
        else:
            pred_answer = str(entry.get("pred_answer", "")).strip().lower()
            if not pred_answer or pred_answer == "none":
                failed_extractions += 1

        if pred_answer == gt_answer:
            correct += 1

    accuracy = (correct / total) * 100 if total > 0 else 0

    print(f"Total Samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Failed extraction of tags count: {failed_extractions}")
    print(f"Final acc (Accuracy): {accuracy:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    target_file = "./src/RP_Inference_and_rate/Generalization/ActivityNetQA/infer_result/ActivityNet-QA_Inference-7b-instruct.json"
    calculate_accuracy(target_file)