import matplotlib


# Non-interactive matplotlib backend. This is required for unit testing where the plot cannot
# actually be shown anywhere. Instead, create them in the non-interactive backend. This function
# is called before any tests are ran. This is neccecary because you cannot change the backend
# after it has been chosen already.
def pytest_configure(config):
    matplotlib.use('Agg')
