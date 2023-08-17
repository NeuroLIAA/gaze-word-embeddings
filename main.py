

def load_corpus(path):
    files = [f for f in path.iterdir()]
    corpus = []
    for file in files:
        if file.is_file():
            with file.open('r') as f:
                corpus.append(f.read())
        elif file.is_dir():
            corpus += load_corpus(file)

    return corpus
