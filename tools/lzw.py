import sys, os

import mybeam

def lzw_encode(string: str) -> list:
    '''
    Encode a string to a list of output symbols.
    
    Example: >>> lzw_encode("A simple test.")
                    [65, 32, 115, 105, 109, 112, 108,
                    101, 32, 116, 101, 115, 116, 46]
    
    :param string: The string to encode.
    
    :return: A list of encoded symbols.
    
    '''
      
    dictionary = dict()
    # Populate the dictionary with regular ASCII-symbols.
    pointer = 256
    for i in range(pointer):
        dictionary[chr(i)] = i
        
    x = ''
    result = []
    for s in string:
        xs = x+s
        if xs in dictionary:
            x = xs
        else:
            result.append(dictionary[x])
            dictionary[xs] = pointer
            pointer += 1
            x = s           
    if x:
        # leftover
        result.append(dictionary[x])
    return result

def lzw_decode(encoded_symbols: list) -> str: 
    '''
    Decompresses a list of output symbols.
    
    :param encoded_symbols: List of encoded symbols.
    
    :return: A string containing the decoded symbols.
    '''
    from io import StringIO
    
    dictionary = dict()
    # Populate the dictionary with regular ASCII-symbols.
    pointer = 256
    for i in range(pointer):
        dictionary[i] = chr(i)
        
    result = StringIO()
    x = chr(encoded_symbols.pop(0))
    result.write(x)
    for c in encoded_symbols:
        if c in dictionary:
            entry = dictionary[c]
        elif c == pointer:
            entry = x + x[0]
        else:
            raise ValueError('Error on key: %s' % c)
        result.write(entry)
        
        dictionary[pointer] = x + entry[0]
        pointer += 1
        
        x = entry
    return result.getvalue()
    
def compress_to_file(encoded: list, filename: str):
    '''
    Packs the encoded symbol-list to a file, 2 bytes at a time.
    
    :param encoded: A list of encoded symbols.
    :param filename: Name of file to pack encoded symbols to.
    '''
    from struct import pack
    
    with open ('{0}'.format(filename), 'wb') as dmp:
        for data in encoded:
            dmp.write(pack('>H', int(data)))
        
def file_to_decompressed(filename: str) -> str:
    '''
    Unpacks the encoded symbol-list from a file and returns
    the decoded string.
    
    :param filename: Name of file containing packed data
    of encoded symbols.
    '''
    from struct import unpack
    
    symbols = []
    file = open('{0}'.format(filename), 'rb')
    while True:
        source = file.read(2)
        if len(source) != 2:
            break;
        else:
            (symbol, ) = unpack('>H', source)
            symbols.append(symbol)
    file.close()
    return lzw_decode(symbols)
    
if __name__ == "__main__":
    import sys
    from markov import *

    path = os.path.dirname(os.path.abspath(__file__))
    HTML = html_to_text(read_html(path+'/Folktale.html'))
    lzw_encoded = lzw_encode(HTML)
    compress_to_file(lzw_encoded, path+'/compressed')
    decompressed = file_to_decompressed(path+'/compressed')
    if decompressed == HTML:
        file = os.stat(path+'/compressed')
        print("Compression-rate: ", 1-os.stat(path+'/compressed').st_size/len(HTML))
        print("File-size:", os.stat(path+'/compressed').st_size, "Bytes")