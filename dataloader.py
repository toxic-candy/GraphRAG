
def load_high(datapath):
    """Read a text file and return its full content as a single string."""
    with open(datapath, 'r', encoding='utf-8') as file:
        return file.read()
