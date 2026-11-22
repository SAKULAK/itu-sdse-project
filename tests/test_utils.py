
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from bsops_the_final import utils


class TestCountClasses(unittest.TestCase):

    def test_count_classes(self):
        labels = [0, 1, 1, 0, 1, 1]

        result = utils.count_classes(labels)

        self.assertEqual(result, {0: 2, 1: 4})


class TestPrintSectionHeader(unittest.TestCase):

    def test_print_section_header(self):
        output = io.StringIO()

        with redirect_stdout(output):
            utils.print_section_header("DATA LOADING")

        expected = (
            "\n"
            + "=" * 50
            + "\nDATA LOADING\n"
            + "=" * 50
            + "\n"
        )

        self.assertEqual(output.getvalue(), expected)


class TestCalculateImageStatistics(unittest.TestCase):

    def test_calculate_image_statistics(self):
        # Two RGB images, each containing two pixels
        images = torch.tensor([
            [
                [[0.0, 1.0]],
                [[0.2, 0.2]],
                [[1.0, 1.0]],
            ],
            [
                [[1.0, 1.0]],
                [[0.8, 0.8]],
                [[0.0, 0.0]],
            ],
        ], dtype=torch.float32)

        labels = torch.tensor([0, 1])

        # Simulate one batch from a DataLoader
        data_loader = [(images, labels)]

        with redirect_stdout(io.StringIO()):
            result = utils.calculate_image_statistics(data_loader)

        self.assertEqual(result["total_samples"], 2)

        expected_mean = [0.75, 0.5, 0.5]
        expected_std = [0.25, 0.0, 0.0]

        for actual, expected in zip(result["mean"], expected_mean):
            self.assertAlmostEqual(actual, expected, places=6)

        for actual, expected in zip(result["std"], expected_std):
            self.assertAlmostEqual(actual, expected, places=6)


class TestSaveClassNames(unittest.TestCase):

    def test_save_class_names(self):
        with TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "artifacts" / "class_names.json"

            class_names = {
                0: "good",
                1: "defective",
            }

            with patch.object(
                utils, "class_names_path", str(output_path)
            ):
                with redirect_stdout(io.StringIO()):
                    utils.save_class_names(class_names)

            self.assertTrue(output_path.exists())

            with output_path.open("r", encoding="utf-8") as f:
                saved_data = json.load(f)

            # JSON converts integer dictionary keys to strings
            self.assertEqual(
                saved_data,
                {
                    "0": "good",
                    "1": "defective",
                },
            )


if __name__ == "__main__":
    unittest.main()
