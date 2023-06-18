from library.load_dataset import generator


def _yield_splitters(paragraphs, changes):
    index = 0
    while index < len(paragraphs) - 1:
        text1 = paragraphs[index]
        text2 = paragraphs[index + 1]
        has_change = changes[index]
        yield {"text1": text1, "text2": text2, "has_change": has_change}
        index += 1


def _preprocess(split):
    preprocessed = list(generator(split)())
    for item in preprocessed:
        text = item["text"]
        changes = item["changes"]
        paragraphs = text.split('\n')
        yield from _yield_splitters(paragraphs, changes)


def preprocess_generator(split="train"):
    def inner_function():
        yield from _preprocess(split)

    return inner_function


if __name__ == "__main__":
    gwag = list(preprocess_generator("train")())
    print(len(gwag))



