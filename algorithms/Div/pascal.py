def pascal_tree(n, base=1):

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
        
if __name__ == '__main__':

    print(pascal_tree(7))