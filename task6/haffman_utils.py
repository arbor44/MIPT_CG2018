from queue import PriorityQueue
from collections import Counter

class Node:
    def __init__(self, left, right, data=None):
        self.left = left
        self.right = right
        self.data = data
        
    def __lt__(self, other):
        return isinstance(other, Node)

    def __gt__(self, other):
        return not self < other

        
def make_tree(words_amount):
    """
    words_amount: dict with key -- word, value -- number of words in input file
    
    returns: root -- root of built tree
    """
    queue_ = PriorityQueue()
    for key, value in words_amount.items():
        queue_.put((value, key))
    
    while queue_.qsize() > 1:
        (value_1, key_1), (value_2, key_2) = queue_.get(), queue_.get()
        queue_.put((value_1 + value_2, Node(key_2, key_1)))
    
    root = queue_.get()
    return root[1]


def haffman_coder(node, prefix, codes={}):
    
    if isinstance(node, Node):
        haffman_coder(node.right, prefix + '0', codes)
        haffman_coder(node.left, prefix + '1', codes)
    else:
        codes[node] = prefix


def bits_to_bytes(bits):
    padding = bits + '0' * ((8 - len(bits)%8)%8)
    ints = [int(padding[i : i + 8], 2) for i in range(0, len(padding), 8)]
    
    return bytes(ints), (8 - len(bits)%8)%8
    

def bytes_to_bits(file):
    bits = ''
    for b in file:
        bit = str(bin(b)[2:])
        bits = bits + '0'*((8-len(bit))%8) + bit
    
    return bits


def translate_to_haffman(input_file, codes):
    encoded_text = ''
    for word in input_file:
        encoded_text = encoded_text + codes[word]
    
    return bits_to_bytes(encoded_text)


def encode(input_file):
    root = make_tree(Counter(input_file))
    codes = {}
    haffman_coder(root, '', codes)
    
    encoded_file, root.data = translate_to_haffman(input_file, codes)
    
    return encoded_file, root


def decode(encoded_file, root):
    output_file = []
    bits = bytes_to_bits(encoded_file)
    bits = bits[:len(bits)-root.data]
    current_node = root
    for b in bits:
        if b == '0':
            current_node = current_node.right
        else:
            current_node = current_node.left

        if isinstance(current_node, Node) == False:
            output_file.append(current_node)
            current_node = root
    
    return bytes(output_file)