import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from standalone_feature_extraction import generate_features_dataframe
from standalone_variables import ENSEMBLE_MODELS_DICTIONARY, MODELS_DICTIONARY


class StandaloneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        example = pd.read_csv(ROOT / "standalone_example.csv")
        library = pd.read_csv(
            ROOT / "resources/data/almanac_cell_lines_library.csv"
        )
        valid = set()
        for column in ("GDSC", "NCI-60", "TISSUE"):
            if column in library.columns:
                valid.update(
                    library[column].dropna().astype(str).str.strip()
                )
        cells = example["Cell"].astype(str).str.strip()
        selected = example.loc[cells.isin(valid)]
        if selected.empty:
            raise RuntimeError("No valid example row was found")
        cls.row = selected.iloc[[0]]

    def test_feature_matrix(self):
        features, dropped = generate_features_dataframe(
            self.row, verbose=False
        )
        values = (
            features.apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        self.assertEqual(features.shape, (1, 1347))
        self.assertEqual(dropped, [])
        self.assertTrue(np.isfinite(values).all())

    @unittest.skipUnless(
        os.environ.get("SYNPRED_MODEL_DIR"),
        "SYNPRED_MODEL_DIR is not configured",
    )
    def test_model_bundle(self):
        individual = [
            Path(path)
            for group in MODELS_DICTIONARY.values()
            for path in group
        ]
        ensembles = [
            Path(path)
            for path in ENSEMBLE_MODELS_DICTIONARY.values()
        ]
        ml = [path for path in individual if path.suffix == ".pkl"]
        dl = [path for path in individual if path.suffix == ".h5"]
        self.assertEqual((len(ml), len(dl), len(ensembles)), (36, 24, 6))
        self.assertEqual(
            [str(path) for path in individual + ensembles if not path.is_file()],
            [],
        )

    @unittest.skipUnless(
        os.environ.get("SYNPRED_TEST_OUTPUT"),
        "SYNPRED_TEST_OUTPUT is not configured",
    )
    def test_end_to_end_output(self):
        output = Path(os.environ["SYNPRED_TEST_OUTPUT"])
        self.assertTrue(output.is_file())
        table = pd.read_csv(output)
        required = [
            "Cell line", "Drug1", "Drug2", "Full-agreement",
            "Bliss", "ZIP", "HSA", "Loewe", "CSS", "Synergy Votes",
        ]
        self.assertEqual(
            [column for column in required if column not in table.columns],
            [],
        )
        self.assertEqual(len(table), 1)
        for column in required[3:]:
            values = pd.to_numeric(table[column], errors="coerce")
            self.assertTrue(np.isfinite(values).all(), column)
        votes = int(table.iloc[0]["Synergy Votes"])
        self.assertTrue(0 <= votes <= 6)


if __name__ == "__main__":
    unittest.main()
