import csv
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, call, patch

from bsops_the_final import data


class TestGetS3KeyFromUrl(unittest.TestCase):

    def test_get_s3_key_from_url(self):
        url = (
            "https://itudsip.hel1.your-objectstorage.com/"
            "data/images/metadata.csv"
        )

        result = data._get_s3_key_from_url(url)

        self.assertEqual(result, "images/metadata.csv")


class TestLoadImageDataset(unittest.TestCase):

    def test_load_valid_images_and_skip_missing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "metadata.csv"
            image_dir = root / "images"
            image_dir.mkdir()

            metadata.write_text(
                "filename,class\n"
                "image1.jpg,0\n"
                "image2.jpg,1\n",
                encoding="utf-8",
            )

            # Only the first image exists
            (image_dir / "image1.jpg").touch()

            with patch.object(data, "metadata_path", str(metadata)):
                with patch.object(data, "image_data_dir", str(image_dir)):
                    with redirect_stdout(io.StringIO()):
                        result = data.load_image_dataset()

            self.assertEqual(result, [
                (image_dir / "image1.jpg", 0)
            ])

    def test_missing_metadata(self):
        with TemporaryDirectory() as tmp:
            missing_path = Path(tmp) / "missing.csv"

            with patch.object(data, "metadata_path", str(missing_path)):
                with self.assertRaises(FileNotFoundError):
                    data.load_image_dataset()


class TestSetupS3Client(unittest.TestCase):

    @patch("bsops_the_final.data.boto3.client")
    def test_setup_s3_client(self, mock_boto_client):
        fake_client = MagicMock()
        mock_boto_client.return_value = fake_client

        result = data._setup_s3_client()

        self.assertIs(result, fake_client)
        mock_boto_client.assert_called_once()

        args, kwargs = mock_boto_client.call_args

        self.assertEqual(args[0], "s3")
        self.assertEqual(kwargs["endpoint_url"], data.S3_ENDPOINT)
        self.assertEqual(
            kwargs["config"].signature_version,
            data.UNSIGNED,
        )


class TestDownloadFile(unittest.TestCase):

    def test_skip_existing_file(self):
        with TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "existing.jpg"
            output_path.touch()

            fake_client = MagicMock()

            result = data._download_file_with_boto3(
                fake_client, "images/raw/existing.jpg", output_path
            )

            self.assertFalse(result)
            fake_client.download_file.assert_not_called()

    def test_successful_download(self):
        with TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "image.jpg"
            fake_client = MagicMock()

            result = data._download_file_with_boto3(
                fake_client, "images/raw/image.jpg", output_path
            )

            self.assertTrue(result)
            fake_client.download_file.assert_called_once_with(
                data.S3_BUCKET,
                "images/raw/image.jpg",
                str(output_path),
            )

    def test_failed_download(self):
        with TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "image.jpg"
            fake_client = MagicMock()
            fake_client.download_file.side_effect = RuntimeError(
                "Connection failed"
            )

            with redirect_stdout(io.StringIO()):
                result = data._download_file_with_boto3(
                    fake_client, "images/raw/image.jpg", output_path
                )

            self.assertFalse(result)


class TestDownloadWithDVC(unittest.TestCase):

    @patch("bsops_the_final.data._setup_s3_client")
    def test_download_with_dvc(self, mock_setup):
        fake_client = MagicMock()
        mock_setup.return_value = fake_client

        with TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "image.jpg"
            url = (
                f"{data.S3_ENDPOINT}/{data.S3_BUCKET}/"
                "images/raw/image.jpg"
            )

            with redirect_stdout(io.StringIO()):
                data.download_with_dvc(url, output_path)

            fake_client.download_file.assert_called_once_with(
                data.S3_BUCKET,
                "images/raw/image.jpg",
                str(output_path),
            )

    @patch("bsops_the_final.data._setup_s3_client")
    def test_skip_existing_file(self, mock_setup):
        fake_client = MagicMock()
        mock_setup.return_value = fake_client

        with TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "existing.jpg"
            output_path.touch()

            with redirect_stdout(io.StringIO()):
                data.download_with_dvc(
                    f"{data.S3_ENDPOINT}/{data.S3_BUCKET}/"
                    "images/raw/existing.jpg",
                    output_path,
                )

            fake_client.download_file.assert_not_called()


class TestDownloadFilesFromMetadata(unittest.TestCase):

    @patch("bsops_the_final.data._setup_s3_client")
    @patch(
        "bsops_the_final.data._download_file_with_boto3",
        return_value=True,
    )
    def test_download_files_from_metadata(
        self, mock_download, mock_setup
    ):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "metadata.csv"
            output_dir = root / "images"

            with metadata.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "class"])
                writer.writerow(["image1.jpg", 0])
                writer.writerow(["image2.jpg", 1])

            with redirect_stdout(io.StringIO()) as output:
                data.download_files_from_metadata(
                    metadata, output_dir, "images/raw/"
                )

            self.assertTrue(output_dir.exists())
            self.assertEqual(mock_download.call_count, 2)

            client = mock_setup.return_value

            mock_download.assert_has_calls(
                [
                    call(
                        client,
                        "images/raw/image1.jpg",
                        output_dir / "image1.jpg",
                    ),
                    call(
                        client,
                        "images/raw/image2.jpg",
                        output_dir / "image2.jpg",
                    ),
                ],
                any_order=True,
            )

            self.assertIn(
                "Downloaded 2 images",
                output.getvalue(),
            )


class TestPullDataFromDVC(unittest.TestCase):

    @patch("bsops_the_final.data.download_files_from_metadata")
    def test_pull_data_from_dvc(self, mock_download):
        data.pull_data_from_dvc()

        self.assertEqual(mock_download.call_count, 2)
        self.assertEqual(
            mock_download.call_args_list,
            [
                call(
                    data.metadata_path,
                    data.image_data_dir,
                    "images/raw/",
                ),
                call(
                    data.test_metadata_path,
                    data.test_data_dir,
                    "test/",
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
