#!/usr/bin/env python
"""
Wrapper for results file
"""
__author__ = "Sid Mau, minor edits by William Cerny"

import yaml
from yaml import CLoader 

class Result:
    def __init__(self, yamlfile):
 
        with open(yamlfile, 'r') as cfg:
            self.results = yaml.load(cfg, Loader = CLoader)['results']
        self.name = self.results['iau']
        texname = 'DELVE {}'.format(self.name)
        texname = texname.replace('-', '$-$')
        texname = texname.replace('+', '$+$')
        self.texname = texname
        self.constellation = self.results['constellation']

    def val(self, key):
        try:
            return self.results[key][0]
        except:
            return self.results[key]

    def err_upper(self, key):
        return self.results[key][1][1] - self.results[key][0]

    def err_lower(self, key):
        return self.results[key][0] - self.results[key][1][0]


