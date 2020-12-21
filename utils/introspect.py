import functools
# Try to import introbind and replace if fail
try:
    import introbind.ib as ib
except:
    class ib():
        def update(name, toAdd):
            return 0

        def start():
            return 0
        
        def stop():
            pass

        def startTrace(name, size):
            return 0

        def stopTrace():
            pass

def introspectTrace(position=None, keyword=None, default=0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            size = default
            if position:
                size = args[position]
            elif keyword:
                size = kwargs.get(keyword, default)
            flag = ib.startTrace(func.__name__, size)
            result = func(*args, **kwargs)
            if flag:
                ib.stopTrace()
            return result
        return wrapper
    return decorator

def introspect(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        ib.update(func.__name__, 1)
        return result
    return wrapper
