import sys
import os
import random
from collections import Counter


def read_html(html_file: str) -> str:
    '''
    Reads a HTML-file and returns its contents.
    
    :param html_file: Name of the file to read.
    
    :return: HTML as a string.
    '''
    with open(html_file, encoding='UTF-8') as f:
        html = f.read()
    return html
    
    
def html_to_text(html: str) -> str:
    '''
    Removes symbols associated with HTML (<,=,>) and returns
    the remaining symbols.
    
    :param html: The html-string to be converted
    
    :return: The remaining text without HTML-tags.
    '''
    return ' '.join([elem for elem in html.replace('\n',' ').split(' ')
    if not any(c in elem for c in '<=>')]).strip()
    
    
def make_rule(text: str, scope=1) -> dict:
    '''
    Creates a markov chain rules-dictionary. The dictionary
    has for keys consecutive symbols appearing in text, of
    length equal to scope, with corresponding values
    all immediately following symbols.
    
    Example: >>> make_rule("abbac")
                {'a': ['b', 'c'], 'b': ['b', 'a']}
                
    Example: >>> make_rule("abbac", scope=2)
                {'ab': ['b'], 'bb': ['a'], 'ba': ['c']}        
    
    :param text: The text to create a dictionary from.
    :param [, scope]: Number of consecutive symbols as keys,
    defaults to scope=1
    
    :return: The rule-dictionary.
    '''
    characters = list(text)
    index = scope
    rule = {}
    
    for char in characters[index:]:
        key = ''.join(characters[index-scope:index])
        if key in rule:
            rule[key].append(char)
        else:
            rule[key] = [char]
        index += 1
    return rule
   
   
def markov(rule: dict, length: int, seed=None) -> str:
    '''
    Generates a random text from a markov chain.
    
    :param rule: A markov chain rule-dictionary
    :param length: The length of the generated text
    :param [, seed]: Random seed. If given will seed the random
    generator.
    
    :return: A string containing the generated text
    '''
    if seed:
        random.seed(seed)

    #Starts with a random set of characters.
    prev_char = random.choice(list(rule.keys()))
    string = ''.join(prev_char)
    
    for i in range(length):
        try:
            key = prev_char
            new_char = random.choice(rule[key])
            string += new_char
            prev_char = (prev_char+new_char)[1:]
        except KeyError:
            return string
    
    return string
    
def entropy(text: str) -> float:
    from collections import Counter
    from math import log2
    '''
    Calculates the entropy of a string.
    
    :param text: The text to calculate
    the entropy of.
    
    :return: entropy
    '''
    counter = Counter(text)
    tot = sum(counter.values())
    pmf = [counter[e]/tot for e in counter]
    return -sum([p*log2(p) for p in pmf])
        
    
if __name__ == '__main__':
    try:
        html = read_html(sys.argv[1])
    except IndexError:
        path = os.path.dirname(os.path.abspath(__file__))
        html = read_html(path+'\Folktale.html')
    source_text = html_to_text(html).lower()
    try:
        rule = make_rule(source_text, int(sys.argv[3]))
    except IndexError:
        rule = make_rule(source_text, 3)
    try:
        my_string = markov(rule, int(sys.argv[2]))
    except IndexError:
        my_string = markov(rule, 1000)
    print(my_string)