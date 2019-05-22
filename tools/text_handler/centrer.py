def center_line(line, length=80):
    """Returns a centered copy of a line. The length-specifier decides
    the length of the full line. Remaining space on full line from
    the inputted line will be filled with spaces."""
    return f"{line:^{length}}"

def center_text(text, length=80):
    """Takes a multi-line text and returns a centered copy. The specified
    length determines the space available on each line - defaults to 80."""
    words = text.split()
    line = ''
    list_lines = list()
    final_text = ''
    curr_line_length = 0

    for word in words:
        if curr_line_length + len(word) < length:
            line = line+' '+word
            curr_line_length += len(word) + 1 # +1 due to the space
        else:
            list_lines.append(center_line(line, length))
            curr_line_length = len(word)
            line = word

    list_lines.append(center_line(line, length)) # Tailing remainder

    for line in list_lines:
        final_text += line+'\n'

    return final_text

if __name__ == '__main__':
    example_line = "This is an example."
    print("Below is a centered line:\n")
    print(center_line(example_line, length=40))
    
    wall_of_text = """This is a huge wall of text, spanding lines and
    lines and lines. Hopefully. The rest now will just be gibberish:
    kjsand akjsd lkand iops ka do dasd  da;a da dn ldad a udak asd
    kjad  kjadoa sda jd;a dd kjaha kahdj  lakda hdadalkd ad lkashd
    lkahd hkd akd ad lkahdlka alk adalhkda ahd alkd had hklad lha
    hdjsah dalkhd sahdka hsakdj haskd jad hsakd hsakj dahk a hda"""


    print("\nAnd below is a centered wall of text:\n")
    print(center_text(wall_of_text, length=40))

