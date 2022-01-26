def search(dictionary, substr):
    for key in dictionary:
        if substr in key:
            return True
    return False
