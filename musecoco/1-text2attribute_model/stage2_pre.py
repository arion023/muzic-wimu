import json
import pickle
import argparse
import os
from copy import deepcopy

def main():
    parser = argparse.ArgumentParser(description="Przetwarzanie wyników atrybutów MuseCoco.")
    
    parser.add_argument('--test_file', type=str, required=True, help='Ścieżka do pliku predict.json (test)')
    parser.add_argument('--predictions', type=str, required=True, help='Ścieżka do pliku predict_attributes.json')
    parser.add_argument('--probabilites', type=str, required=True, help='Ścieżka do pliku softmax_probs.json')
    parser.add_argument('--attributes', type=str, required=True, help='Ścieżka do pliku att_key.json')
    parser.add_argument('--output_file', type=str, default='infer_test.bin', help='Nazwa pliku wyjściowego (.bin)')

    args = parser.parse_args()

    test = json.load(open(args.test_file, 'r'))
    pred = json.load(open(args.predictions, 'r'))
    probs = json.load(open(args.probabilites, 'r'))
    att_key = json.load(open(args.attributes, 'r'))

    final = []
    for line in test:
        ins = {}
        ins['text'] = line['text']
        ins['pred_labels'] = {}
        ins['pred_probs'] = {}
        final.append(deepcopy(ins))

    for k, v in pred.items():
        for j in range(len(v)):
            final[j]['pred_labels'][k] = deepcopy(v[j])
            
    for k, v in probs.items():
        for j in range(len(v)):
            final[j]['pred_probs'][k] = deepcopy(v[j])

    # Logika filtrowania kluczy
    I1s2_key = [att for att in att_key if att.startswith("I1s2")]
    S4_key = [att for att in att_key if att.startswith("S4")]

    for idx in range(len(final)):
        pred_labels_I1s2, pred_probs_I1s2 = [], []
        pred_labels_S4, pred_probs_S4 = [], []
        
        for i1s2 in I1s2_key:
            pred_labels_I1s2.append(deepcopy(final[idx]['pred_labels'][i1s2]))
            pred_probs_I1s2.append(deepcopy(final[idx]['pred_probs'][i1s2]))
            final[idx]['pred_labels'].pop(i1s2)
            final[idx]['pred_probs'].pop(i1s2)
            
        for s4 in S4_key:
            pred_labels_S4.append(deepcopy(final[idx]['pred_labels'][s4]))
            pred_probs_S4.append(deepcopy(final[idx]['pred_probs'][s4]))
            final[idx]['pred_labels'].pop(s4)
            final[idx]['pred_probs'].pop(s4)
            
        final[idx]['pred_probs']['I1s2'] = pred_probs_I1s2
        final[idx]['pred_probs']['S4'] = pred_probs_S4
        final[idx]['pred_labels']['I1s2'] = pred_labels_I1s2
        final[idx]['pred_labels']['S4'] = pred_labels_S4

    path = args.output_file
    splitted = path.split("/")
    dir_path = "/".join(splitted[:-1])
    if not os.path.exists(dir_path):
        print("making... ", dir_path)
        os.makedirs(dir_path)

    with open(args.output_file, 'wb') as f:
        pickle.dump(final, f)
    
    print(f"Zakończono! Wynik zapisano w: {args.output_file}")

if __name__ == "__main__":
    main()