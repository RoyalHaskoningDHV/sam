import warnings


def parametrized(dec):
    """Decorator to make a decorator parametrized"""

    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)

        return repl

    return layer


@parametrized
def add_future_warning(func, msg):
    """Decorator to add a FutureWarning to a function."""

    def wrapper(*args, **kwargs):
        warnings.warn(msg, FutureWarning)
        return func(*args, **kwargs)

    return wrapper
