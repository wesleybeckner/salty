from __future__ import absolute_import, division, print_function
import salty
from rdkit.Chem import AllChem as Chem
import unittest
import datetime


class iupac_smiles_tests(unittest.TestCase):
    data_files = ["cationInfo.csv", "anionInfo.csv"]
    df = salty.load_data(data_files[0])
    smiles = df.smiles
    for i in range(len(smiles)):
        ion = smiles[i]
        salty.check_name(ion)

    def test_1_check_data(self):
        for i in range(len(self.data_files)):
            df = salty.load_data(self.data_files[i])
            self.check_data(df)

    def test_2_check_wrong_ion(selfs):
        ion = 'stupid_nonsense_string'
        salty.check_name(ion)

    def test_benchmark(self):
        salty.Benchmark.run(self.test_1_check_data)
        salty.Benchmark.run(self.test_2_check_wrong_ion)

    def check_data(self, df):
        startTime = datetime.datetime.now()

        def fnDisplay(message):
            display(message, startTime)

        smiles = df.smiles
        for i in range(len(smiles)):
            ion = smiles[i]
            try:
                Chem.SanitizeMol(Chem.MolFromSmiles(ion))
            except ValueError:
                name = salty.check_name(ion)
                message = "RDKit cannot interpret %s ion SMILES in datafile" \
                          % name
                fnDisplay(message)
            if "-" not in ion and "+" not in ion:
                name = salty.check_name(ion)
                message = "%s ion does not have a charge" % name
                fnDisplay(message)
            if "." in ion:
                name = salty.check_name(ion)
                message = "%s ion contains more than one molecular entity" \
                          % name
                fnDisplay(message)


def display(message, startTime):
    timeDiff = datetime.datetime.now() - startTime
    print("{}\t{}".format(timeDiff, message))


if __name__ == '__main__':
    unittest.main()
