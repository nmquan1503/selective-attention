def reset_buffers(names):
    return lambda nargs: [
        nargs[name].zero_() 
        for name in names 
        if nargs[name] is not None
    ]