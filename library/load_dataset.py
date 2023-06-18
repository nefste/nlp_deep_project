import os
import json


def _generator_per_set(dataset):
    file_count = sum(1 for entry in os.scandir(dataset) if entry.is_file())
    print(file_count)
    amount = int(file_count / 2)
    print("Found " + str(amount) + " of examples in the dataset", dataset)
    for i in range(1, amount):
        json_file_path = os.path.join(dataset, "truth-problem-" + str(i) + ".json")
        text_file_path = os.path.join(dataset, "problem-" + str(i) + ".txt")
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        with open(text_file_path, 'r') as f_text:
            text = f_text.read()
            paragraphs = text.split('\n')
        if len(paragraphs) - 1 != len(data["changes"]):
            print("MAN THERE was an error at file " + str(i))
        else:
            yield {"text": text, "changes": data["changes"], "paragraph-authors": data["paragraph-authors"]}


def _generator(split):
    directory = '../data/dataset'
    for i in range(1, 4):
        dataset = directory + str(i) + '/' + split
        yield from _generator_per_set(dataset)


def generator(split="train"):
    def inner_function():
        yield from _generator(split)

    return inner_function
