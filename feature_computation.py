def n_grams(word, n = 3, padding = True):
    if padding:
        pads = "#" * (n - 1)
        word = pads + word + pads
    ngrams = []
    for i in range(0, len(word) - n + 1):
        ngrams.append(word[i:(i+n)])
    return ngrams

def starts_with_uppercase(word):
    return word[0].lower() != word[0]
