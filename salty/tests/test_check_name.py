from __future__ import absolute_import, division, print_function
import salty
# uncomment for local tests
# import os
# import sys
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#         sys.path.append(module_path)
# import localsalty as salty
from rdkit.Chem import AllChem as Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import\
    MolecularDescriptorCalculator as calculator
import numpy as np
import unittest
import datetime
from math import exp
import random


class check_data_tests(unittest.TestCase):
    data_files = ["cationInfo.csv", "anionInfo.csv"]
    
    def test_1_check_data(self):
        for i in range(len(self.data_files)):
            df = salty.load_data(self.data_files[i])
            self.check_data(df)

    def test_benchmark(self):
        salty.Benchmark.run(self.test_1_check_data)

    def check_data(self, df):
        startTime = datetime.datetime.now()

        def fnDisplay(message):
            display(message, startTime)

        smiles = df.smiles
        for i in range(len(smiles)):
            ion = smiles[i]
            try:
                Chem.SanitizeMol(Chem.MolFromSmiles(ion))
            except ArgumentError:
                name = salty.checkName(ion)
                message = "RDKit cannot interpret %s ion SMILES in datafile" % name
                fnDisplay(message)
            if "-" not in ion and "+" not in ion:
                name = salty.checkName(ion)
                message = "%s ion does not have a charge" % name
                fnDisplay(message)
            if "." in ion:
                name = salty.checkName(ion)
                message = "%s ion contains more than one molecular entity" % name
                fnDisplay(message)


def display(message, startTime):
    timeDiff = datetime.datetime.now() - startTime
    print("{}\t{}".format(timeDiff, message))


if __name__ == '__main__':
    unittest.main()
