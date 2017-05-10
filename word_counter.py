from corpus_reader import read_dataset
import re

if __name__ == "__main__":
    cnt = {}
    data = read_dataset()
    for document in data:
        for sentence in document.sentences:
            if len(sentence.pairs) > 0:
                for word in re.split("\W+", sentence.text):
                    if len(word) > 0:
                        if word not in cnt:
                            cnt[word] = 1
                        else:
                            cnt[word] = cnt[word] + 1
    srt = sorted([(cnt[word], word) for word in cnt], reverse = True)
    for pair in srt:
        print(pair[1], pair[0])
