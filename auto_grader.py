'''
Main script for testing the assignment.
Runs the tests on the results json file.
'''

import argparse
import json

def get_args():
    parser = argparse.ArgumentParser(description='Language Modeling')
    parser.add_argument('test', type=str, help='The test to perform.')
    return parser.parse_args()

def test_read_data(results):
    lengths = tuple(results["lengths"])

    if not lengths == (1750, 250, 500):
        return f"Lengths are {lengths}, expected (1750, 250, 500)"
    return 1

def test_vocab(results):
    if results["length"] != 7163:
        return f"Vocab length is {results['vocab_length']}, expected 7163"
    if results["tag2id_length"] not in [7, 8]:
        return f"Number of tags is {results['tag2id_length']}, expected 7"
    if results["Spongebob"] != 1:
        return f"Index of 'Spongebob' is {results['Spongebob']}, expected 1 because it is unknown"
    return 1

def test_count_oov(results):
    if results["dev_oov"] != 638:
        return f"Number of OOV words in dev is {results['dev_oov']}, expected 638"
    if results["test_oov"] != 1368:
        return f"Number of OOV words in test is {results['test_oov']}, expected 1368"
    return 1
    
def test_prepare_data_loader(results):
    if not tuple(results["lengths"]) == (110, 16, 32):
        return f"Lengths are {results['lengths']}, expected (110, 16, 32)"
    return 1

def test_NERNet(results):
    f1 = results["f1"]
    f1_wo_o = results["f1_wo_o"]

    # Min value to pass
    if f1 < 0.80:
        return f"F1 is {f1}, expected at least 0.80"
    if f1_wo_o < 0.60:
        return f"F1 without O is {f1_wo_o}, expected at least 0.60"
    
    # Values to partially pass
    if f1 < 0.88:
        return 2
    if f1_wo_o < 0.66:
        return 2

    # Pass with full marks
    return 1

def test_glove(results):
    f1 = results["f1"]
    f1_wo_o = results["f1_wo_o"]

    # Min value to pass
    if f1 < 0.80:
        return f"F1 is {f1}, expected at least 0.80"
    if f1_wo_o < 0.60:
        return f"F1 without O is {f1_wo_o}, expected at least 0.60"
    
    # Values to partially pass
    if f1 < 0.85:
        return 2
    if f1_wo_o < 0.64:
        return 2

    # Pass with full marks
    return 1

def main():
    # Get command line arguments
    args = get_args()

    # Read results.json
    with open('results.json', 'r') as f:
        results = json.load(f)

    # Initialize the result variable
    result = None

    # Switch between the tests
    match args.test:
        case 'test_read_data':
            result = test_read_data(results["test_read_data"])
        case 'test_vocab':
            result = test_vocab(results["test_vocab"])
        case 'test_count_oov':
            result = test_count_oov(results["test_count_oov"])
        case 'test_prepare_data_loader':
            result = test_prepare_data_loader(results["test_prepare_data_loader"])
        case 'test_NERNet':
            result = test_NERNet(results["test_NERNet"])
        case 'test_glove':
            result = test_glove(results["test_glove"])
        case _:
            print('Invalid test.')

    # Print the result for the autograder to capture
    if result is not None:
        print(result)

if __name__ == '__main__':
    main()
