class CharRemover:
    def __init__(self):
        self.chars = ['—', '‒', '−', '-', '«', '»', '“', '”', '\'', '\"', '‘', '’', '(', ')', ';', ',', ':', '.', '…', '¿', '?', '¡', '!', '=']

    def remove(self, text):
        return ''.join(c for c in text if c not in self.chars)