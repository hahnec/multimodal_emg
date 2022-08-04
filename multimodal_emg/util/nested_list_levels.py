def level_enumerate(data, level=1):
    result = []
    for elem in data:
        if isinstance(elem, (list, tuple)):
             result.extend(level_enumerate(elem, level + 1))
        else:
             result.append(level)
    return result
