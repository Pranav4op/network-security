import yaml
from networksecurity.exception.exception import NetworkSecurityException
import os, sys
import numpy as np
import dill
import pickle
import logging

logger = logging.getLogger(__name__)


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
