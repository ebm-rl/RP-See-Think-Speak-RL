import json
import re

def calculate_accuracy(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    correct = 0
    total = len(data)
    failed_extractions = 0

    print(f"Start evalute: {json_file}")
    print("-" * 30)

    for entry in data:
        gt_answer = str(entry.get("answer")).strip()
        raw_output = entry.get("model_raw_output", "")
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', raw_output, re.DOTALL)
        
        if match:
            pred_answer = match.group(1).strip()
            if pred_answer == gt_answer:
                correct += 1
        else:
            failed_extractions += 1

    accuracy = (correct / total) * 100 if total > 0 else 0

    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Failed label extraction: {failed_extractions}")
    print(f"Final accuracy (Accuracy): {accuracy:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    target_file = "./src/RP_Inference_and_rate/Generalization/Next-QA/infer_result/NExTQA_CH_Inference-7b-instruct-ALL.json"
    calculate_accuracy(target_file)