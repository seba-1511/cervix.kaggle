#!/usr/bin/env python

from classification import get_classification
from localization import get_localization

problems = {
        'localization': get_localization,
        'classification': get_classification,
}
