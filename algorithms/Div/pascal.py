def pascal_tree(n: int, base:int = 1) -> dict:

    """Makes a dictionary where the keys are row-indexes in a pascal-trangle
    of size n, and the values are the rows as a list. E.g. pascal(3) should
    return {1 : [1], 2: [1,1], 3: [1,2,1]}.
    
    pascal(0) should returns an empty dictionary.
    
    Optional argument 'base=': set an integer as a new base. E.g.
    pascal(3, base=9) should return {1: [2], 2: [2, 2], 3: [2, 4, 2]}"""

    
    for v in (n,base):
        assert isinstance(v, int),f'Both n and base must be integers'
    if not n:
        return {}
    if n == 1:
        return {1: [base]}
    else:
        bottom_row = list()
        prev_p = pascal_tree(n-1, base)  #Only one recursive call
        for i in range(0, n):
            if i == 0:
                bottom_row.append(prev_p[n-1][i])
            elif i == n-1:
                bottom_row.append(prev_p[n-1][i-1])
            else:
                bottom_row.append(prev_p[n-1][i-1]+prev_p[n-1][i])
        bottom_row = {n: bottom_row}
        pascal_dict = prev_p
        pascal_dict.update(bottom_row)
        return pascal_dict
        
def pascal(n: int, method: str = 'memoization') -> int:

    """Calculates pascal number n recursively using memoization,
    or with optional method specified.
    
    :param: n The pascal number to calculate
    
    :param: (Optional) method=['memoization', 'dynamic']
    
    :returns: int m where m is pascal number n.
    """
    
    assert (method == 'memoization' or method == 'dynamic'),\
            ('method parameter can only be either memoization or dynamic.')
    mem = {0: 0, 1: 1, 2: 1}
    if method == 'memoization':
        if n in mem:
            return mem[n]
        else:
            return (pascal_memoized(n-1, mem)
                    + pascal_memoized(n-2, mem))
    elif method == 'dynamic':
        for i in range(3,n+1):
            mem[i] = mem[i-1] + mem[i-2]
        return mem[n]
    
    
def pascal_memoized(n: int, mem: dict) -> int:

    """Helper function taking a storing dictionary as as
    a parameter.
    """

    if n in mem:
        return mem[n]
    else:
        mem[n] = (pascal_memoized(n-1, mem)
                  + pascal_memoized(n-2, mem))
        return mem[n]
        
if __name__ == '__main__':

    print(pascal_tree(7))
    print(pascal(1000, method='memoization'))
    print(pascal(1000, method='dynamic'))