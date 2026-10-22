"""
Pytest configuration for NeuroScript AI test suite.
Initializes TensorFlow runtime prior to any other C++ extension imports (e.g., PyArrow)
to prevent Windows DLL DllMain initialization order conflicts.
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import pyarrow
