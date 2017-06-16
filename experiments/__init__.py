#!/usr/bin/env python

from .classification import get_classification, get_classification_test
from .localization import get_localization

problems = {
    'localization': get_localization,
    'classification': get_classification,
}

tests = {
    'classification': get_classification_test
}
