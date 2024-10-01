import pytest
import requests
from unittest.mock import patch, mock_open
import common
from processor.image_processor import ImageProcessor


class TestImageProcessor:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.image_processor = ImageProcessor("test_image.jpg")

    def test_is_url_local(self):
        assert not self.image_processor.is_url()

    def test_is_url_remote(self):
        self.image_processor.image_path = "http://example.com/image.jpg"
        assert self.image_processor.is_url()

    def test_encode_image_file(self):
        with patch("builtins.open", mock_open(read_data=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")) as mock_file:
            self.image_processor.image_path = "test_image.png"
            self.image_processor.encode_image()
            assert self.image_processor.img_base64 is not None

    def test_encode_image_file_not_found(self):
        self.image_processor.image_path = "non_existent_image.jpg"
        with pytest.raises(FileNotFoundError):
            self.image_processor.encode_image()

    def test_encode_image_file_read_error(self):
        with patch("builtins.open", mock_open()) as mock_file:
            mock_file.side_effect = IOError("Failed to read file")
            with pytest.raises(IOError):
                self.image_processor.encode_image()

    def test_encode_image_url(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value.content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            self.image_processor.image_path = "http://example.com/image.png"
            self.image_processor.encode_image()
            assert self.image_processor.img_base64 is not None

    def test_encode_image_url_failure(self):
        with patch("requests.get", side_effect=requests.RequestException):
            self.image_processor.image_path = "http://example.com/image.png"
            with pytest.raises(IOError):
                self.image_processor.encode_image()

    def test_display_image_base64(self):
        self.image_processor.img_base64 = "test_base64_data"
        with patch("IPython.display.display") as mock_display:
            self.image_processor.display_image_base64()
            mock_display.assert_called_once()

    def test_display_image_base64_error(self):
        self.image_processor.img_base64 = None
        with pytest.raises(ValueError):
            self.image_processor.display_image_base64()
