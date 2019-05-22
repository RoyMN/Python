def center_line("line, length=80"):
    """Returns a centered copy of a line. The length-specifier decides
    the length of the full line. Remaining space on full line from
    the inputted line will be filled with spaces."""
    return f"{line:^{length}}"

