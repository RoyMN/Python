def label_nodes(nodes, label, result, prefix = ''):
    '''
    Helper-function to label each node recursively.
    '''
    
    childs = nodes[label]
    tree = {}
    if len(childs) == 2:
        tree['0'] = label_nodes(nodes, childs[0], result, prefix+'0')
        tree['1'] = label_nodes(nodes, childs[1], result, prefix+'1')     
        return tree
    else: # leaf
        result[label] = prefix
        return label

        
def huff_tree(_vals):
    '''
    Generates a huffman-tree from a list of frequencies.
    
    :param _vals: Frequency-list
    
    :return: Symbols which is a dictionary connecting
    symbols and encoded symbols. And tree which is a
    dictionary representing the huffman-tree.
    '''
    
    # Avoid changing input.
    vals = _vals.copy()
    nodes = {}
    # leafs initialization
    for n in vals.keys():
        nodes[n] = []
       
    while len(vals) > 1:
        s_vals = sorted(vals.items(), key=lambda x:x[1]) 
        a1 = s_vals[0][0]
        a2 = s_vals[1][0]
        vals[a1+a2] = vals.pop(a1) + vals.pop(a2)
        nodes[a1+a2] = [a1, a2]      
        
    # Store dictionary between symbols and encoded symbols.
    symbols = {}
    root = a1+a2
    tree = label_nodes(nodes, root, symbols)
 
    return symbols, tree
    
    
def huff_encode(iter, sym):
    '''
    Generates an encoded string using a iterable
    structure and a dictionary for encoding each symbol.
    
    :param iter: An iterable structure like list or string.
    :param sym: A dictionary connecting symbols and encoded symbols.
    
    :return: String containing the encoded sequence of symbols.
    '''
    
    return ''.join([sym[str(e)] for e in iter])
    
    
def huff_decode(encoded, tree, string=False):  
    '''
    Decodes a huffman-encoded string, given a huffman-tree.
    Defaults to output of a list of decoded symbols unless
    string=True, in which case output will be a string of
    decoded symbols.
    '''
    
    decoded = []
    i = 0
    while i < len(encoded):
        sym = encoded[i]
        label = tree[sym]
        # Continue untill leaf is reached
        while not isinstance(label, str):
            i += 1
            sym = encoded[i]
            label = label[sym]        
        decoded.append(label)        
        i += 1
    if string == True:
        return ''.join([e for e in decoded])
    return decoded
    
        

def freq_dict(l: list) -> dict:
    '''
    Returns a frequency-dictionary {'c':n}
    for c in l where n is the number of times c appears in l.
    
    :param l: list to generate frequency-list dictionary from.
    
    :return: A dictionary connecting items from a list to their
    frequencies.
    '''
    
    dict = {}
    for c in l:
        if not str(c) in dict:
            dict[str(c)] = l.count(c)
    return dict
    
    
if __name__ == '__main__':
    from markov import *
    from lzw import *
    
    # Reading Folktale.html and extracting only the body-text.
    path = os.path.dirname(os.path.abspath(__file__))
    HTML = html_to_text(read_html(path+'/Folktale.html'))
    
    # Encoding with lzw-compression.
    lzw_encoded = lzw_encode(HTML)

    # Encoding with Huffman-compression
    lzw_freq = freq_dict(lzw_encoded)
    symbols, tree = huff_tree(lzw_freq)
    huff_encoded = huff_encode(lzw_encoded, symbols)
   
    # Back-trace with decoding.
    huff_decoded = huff_decode(huff_encoded, tree)
    lzw_decoded = lzw_decode([int(e) for e in huff_decoded])
    if lzw_decoded == HTML:
        print("Plain-text in bits:",len(HTML*8))
        print("lzw-encoding in bits:", len(lzw_encoded)*16,
        ", compression-rate:", 1-(len(lzw_encoded)*16/(len(HTML)*8)))
        print("Huff-encoding in bits:", len(huff_encoded),
        ", compression-rate:", 1-(len(huff_encoded)/(len(HTML)*8)))
    else:
        "Decompression failed."