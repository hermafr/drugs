from parsing_example import get_parser, parse
from corpus_reader import Entity, read_dataset

def to_span(node, isWord = False):
    if not isWord:
        node = node.get_word()
    span = (node.get_span_start(), node.get_span_finish())
    return span

class ParseTreeFeatureExtractor:
    def __init__(self):
        self.parser = get_parser()
    
    def parse(self, sentence):
        return parse(self.parser, sentence)
    
    def get_trees(self, sentence):
        sentence_list = self.parse(sentence)
        return [s.get_dep_tree() for s in sentence_list]
    
    def rec_find(self, node, entity):
        leftmost_pos = entity.char_offset[0][0]
        if node.get_word().get_span_start() <= leftmost_pos and node.get_word().get_span_finish() > leftmost_pos:
            return True, node
        for i in range(node.num_children()):
            found, target_child = self.rec_find(node.nth_child(i), entity)
            if found:
                return True, target_child
        return False, None
    
    def find(self, parsed_sentence, entity):
        for tree in parsed_sentence:
            found, node = self.rec_find(tree.get_dep_tree().begin(), entity)
            if found:
                return True, node
        return False, None
    
    def is_head(self, node, ls):
        for l in ls:
            node_span = to_span(node)
            head_span = to_span(l.get_dep_tree().begin())
            if to_span(node) == to_span(l.get_dep_tree().begin()):
                return True
        return False
    
    def lowest_common_ancestor(self, node1, node2, ls):
        ancestors = set()
        # ancestors of node1:
        node = node1
        while not self.is_head(node, ls):
            ancestors.add(to_span(node))
            node = node.get_parent()
        ancestors.add(to_span(node))  # add head too
        # node2:
        node = node2
        while not self.is_head(node, ls):
            if to_span(node) in ancestors:
                return True, node
            node = node.get_parent()
        if to_span(node) in ancestors:  # check head too
            return True, node
        return False, None

"""
txt = "The beautiful cat that lives here eats tasty fish. The fish tastes good."
parser = ParseTreeFeatureExtractor()
ls = parser.parse(txt)

e1 = Entity("", [(14,17)], "", "cat")
found1 = parser.find(ls, e1)
#print(found1)
print(found1.get_word().get_form())

e2 = Entity("", [(45,49)], "", "fish")
#e2 = Entity("", [(55,59)], "", "fish")
found2 = parser.find(ls, e2)
#print(found2)
print(found2.get_word().get_form())

lca = parser.lowest_common_ancestor(found1, found2, ls)
print("found:")
print(lca)
print(lca.get_word().get_form())
"""

if __name__ == "__main__":
    data = read_dataset()

    parser = ParseTreeFeatureExtractor()

    counters = {}

    for doc in data:
        for sentence in doc.sentences:
            parsed_sentence = parser.parse(sentence.text)
            for pair in sentence.pairs:
                found1, node1 = parser.find(parsed_sentence, pair.e1)
                found2, node2 = parser.find(parsed_sentence, pair.e2)
                if not found1 or not found2:
                    ancestor_word = "not_found"
                    tag = "not_found"
                else:
                    found, lca = parser.lowest_common_ancestor(node1, node2, parsed_sentence)
                    """
                    print(pair.e1.text,
                          pair.e2.text,
                          pair.type,
                          lca.get_word().get_form() if found else None,
                          lca.get_word().get_tag() if found else None)
                    """
                    tag = lca.get_word().get_tag() if found else "null"
                    ancestor_word = lca.get_word().get_form() if found else "null"
                if tag not in counters:
                    counters[tag] = {}
                p_type = pair.type if pair.type is not None else "null"
                if p_type not in counters[tag]:
                    counters[tag][p_type] = 0
                counters[tag][p_type] = counters[tag][p_type] + 1
                print("%s,%s,%s,%s" % (pair.id, p_type, tag, ancestor_word))
    
    for tag in sorted(counters):
        for ty in sorted(counters[tag]):
            print(tag, ty, counters[tag][ty])
