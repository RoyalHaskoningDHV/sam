import os

import matplotlib

# For this unit test, we don't need GPU
# So run on CPU to increase reliability
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# We don't want some of the tensorflow warnings that aren't our fault anyway
# Unfortunately, this only works on some of the more recent versions of tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# Non-interactive matplotlib backend. This is required for unit testing where the plot cannot
# actually be shown anywhere. Instead, create them in the non-interactive backend. This function
# is called before any tests are ran. This is neccecary because you cannot change the backend
# after it has been chosen already.
def pytest_configure(config):
    matplotlib.use("Agg")


# Tensorflow is an optional dependency
try:
    import tensorflow as tf

    # Force tensorflow to run single-threaded, for further determinism
    # This needs to be done immediately after loading tensorflow
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except ImportError:
    pass
