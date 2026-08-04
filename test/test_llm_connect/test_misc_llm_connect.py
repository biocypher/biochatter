"""Tests for the miscellaneous LLM connect module."""

import os
import tempfile
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest
from PIL import Image

from biochatter._image import (
    convert_and_resize_image,
    convert_to_pil_image,
    convert_to_png,
    encode_image,
    encode_image_from_url,
    process_image,
)
from biochatter.llm_connect import (
    OllamaConversation,
    WasmConversation,
)
from biochatter.llm_connect.conversation import Conversation

from langchain_core.messages import AIMessage, HumanMessage


class MockConversation(Conversation):
    """Mock conversation class for testing."""

    def __init__(self):
        super().__init__(model_name="test-model", prompts={})
        self.messages = []

    def set_api_key(self, api_key: str, user: str | None = None) -> None:
        """Mock implementation."""

    def _primary_query(self, **kwargs):
        """Mock implementation."""
        return "test response", {"total_tokens": 100}

    def _correct_response(self, msg: str) -> str:
        """Mock implementation."""
        return "corrected response"


def test_ollama_chatting():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    with patch("biochatter.llm_connect.ollama.ChatOllama") as mock_model:
        response = AIMessage(
            content="Hello there! It's great to meet you!",
            additional_kwargs={},
            response_metadata={
                "model": "llama3",
                "created_at": "2024-06-20T17:19:45.376245476Z",
                "message": {"role": "assistant", "content": ""},
                "done_reason": "stop",
                "done": True,
                "total_duration": 256049685,
                "load_duration": 3096978,
                "prompt_eval_duration": 15784000,
                "eval_count": 11,
                "eval_duration": 107658000,
            },
            type="ai",
            name=None,
            id="run-698c8654-13e6-4bbb-8d59-67e520f78eb3-0",
            example=False,
            tool_calls=[],
            invalid_tool_calls=[],
            usage_metadata=None,
        )

        mock_model.return_value.invoke.return_value = response

        convo = OllamaConversation(
            base_url=base_url,
            model_name="llama3",
            prompts={},
            correct=False,
        )
        (msg, token_usage, correction) = convo.query("Hello, world!")
        assert token_usage > 0


def test_wasm_conversation():
    # Initialize the class
    wasm_convo = WasmConversation(
        model_name="test_model",
        prompts={},
        correct=True,
        split_correction=False,
    )

    # Check if the model_name is correctly set
    assert wasm_convo.model_name == "test_model"

    # Check if the prompts are correctly set
    assert wasm_convo.prompts == {}

    # Check if the correct is correctly set
    assert wasm_convo.correct == True

    # Check if the split_correction is correctly set
    assert wasm_convo.split_correction == False

    # Test the query method
    test_query = "Hello, world!"
    result, _, _ = wasm_convo.query(test_query)
    assert result == test_query  # assuming the messages list is initially empty

    # Test the _primary_query method, add another message to the messages list
    wasm_convo.append_system_message("System message")
    result = wasm_convo._primary_query()
    assert result == test_query + "\nSystem message"


def test_convert_and_resize_image():
    with Image.new("RGB", (2000, 2000)) as img:
        resized_img = convert_and_resize_image(img, max_size=1000)
        assert resized_img.size == (1000, 1000)


def test_convert_to_png():
    with Image.new("RGB", (100, 100)) as img:
        png_data = convert_to_png(img)
        assert isinstance(png_data, bytes)
        assert png_data.startswith(b"\x89PNG")


@patch("biochatter._image.pdf2image.convert_from_path")
def test_convert_to_pil_image_pdf(mock_convert_from_path):
    mock_convert_from_path.return_value = [Image.new("RGB", (1000, 1000))]
    with patch("biochatter._image.os.path.exists", return_value=True):
        with patch(
            "biochatter._image.os.path.abspath",
            side_effect=lambda x: x,
        ):
            img = convert_to_pil_image("test.pdf")
            assert isinstance(img, Image.Image)


@patch("biochatter._image.subprocess.run")
@patch("biochatter._image.os.path.exists", return_value=True)
@patch("biochatter._image.os.path.abspath", side_effect=lambda x: x)
def test_convert_to_pil_image_eps(mock_abspath, mock_exists, mock_run):
    with Image.new("RGB", (1000, 1000)) as img:
        with patch("biochatter._image.Image.open", return_value=img):
            converted_img = convert_to_pil_image("test.eps")
            assert isinstance(converted_img, Image.Image)


@patch("biochatter._image.Image.open")
@patch("biochatter._image.os.path.exists", return_value=True)
@patch("biochatter._image.os.path.abspath", side_effect=lambda x: x)
def test_convert_to_pil_image_unsupported(mock_abspath, mock_exists, mock_open):
    with pytest.raises(ValueError):
        convert_to_pil_image("test.txt")


def test_process_image():
    with patch("biochatter._image.convert_to_pil_image") as mock_convert:
        with Image.new("RGB", (100, 100)) as img:
            mock_convert.return_value = img
            encoded_image = process_image("test.jpg", max_size=1000)
            assert isinstance(encoded_image, str)
            assert encoded_image.startswith("iVBORw0KGgo")  # PNG base64 start


def test_encode_image():
    with Image.new("RGB", (100, 100)) as img:
        m = mock_open(read_data=img.tobytes())
        with patch("builtins.open", m):
            encoded_str = encode_image("test.jpg")
            assert isinstance(encoded_str, str)


def test_encode_image_from_url():
    with patch("biochatter.llm_connect.conversation.urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = b"image_data"
        mock_urlopen.return_value.__enter__.return_value = mock_response
        mock_urlopen.return_value.info.return_value.get_content_type.return_value = "image/jpeg"

        with patch("tempfile.NamedTemporaryFile", new_callable=MagicMock) as mock_tempfile:
            mock_tempfile_instance = mock_tempfile.return_value.__enter__.return_value
            mock_tempfile_instance.name = "test_temp_file"

            write_mock = Mock()
            mock_tempfile_instance.write = write_mock

            with patch("biochatter._image.encode_image") as mock_encode:
                mock_encode.return_value = "base64string"

                with patch("os.remove") as mock_remove:
                    encoded_str = encode_image_from_url(
                        "http://example.com/image.jpg",
                    )

            write_mock.assert_called_once_with(b"image_data")
            mock_remove.assert_called_once_with("test_temp_file")
            assert isinstance(encoded_str, str)
            assert encoded_str == "base64string"


def test_append_single_image_message():
    """Test that append_image_message works with a single image (backward compatibility)."""
    convo = MockConversation()

    # Create a temporary image file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        # Create a simple image and save it
        img = Image.new("RGB", (100, 100), color="red")
        img.save(tmp_file.name, "JPEG")
        tmp_path = tmp_file.name

    try:
        # Test with single image string
        convo.append_image_message("Describe this image", tmp_path, local=True)

        # Verify the message was added correctly
        assert len(convo.messages) == 1
        message = convo.messages[0]
        assert isinstance(message, HumanMessage)
        assert isinstance(message.content, list)
        assert len(message.content) == 2  # text + 1 image

        # Check text content
        text_content = message.content[0]
        assert text_content["type"] == "text"
        assert text_content["text"] == "Describe this image"

        # Check image content
        image_content = message.content[1]
        assert image_content["type"] == "image_url"
        assert "image_url" in image_content
        assert image_content["image_url"]["url"].startswith("data:image/jpeg;base64,")

    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_append_multiple_images_message():
    """Test that append_image_message works with multiple images."""
    convo = MockConversation()

    # Create temporary image files
    tmp_paths = []
    for i in range(3):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            # Create a simple image with different colors
            colors = ["red", "green", "blue"]
            img = Image.new("RGB", (100, 100), color=colors[i])
            img.save(tmp_file.name, "JPEG")
            tmp_paths.append(tmp_file.name)

    try:
        # Test with multiple images
        convo.append_image_message("Compare these images", tmp_paths, local=True)

        # Verify the message was added correctly
        assert len(convo.messages) == 1
        message = convo.messages[0]
        assert isinstance(message, HumanMessage)
        assert isinstance(message.content, list)
        assert len(message.content) == 4  # text + 3 images

        # Check text content
        text_content = message.content[0]
        assert text_content["type"] == "text"
        assert text_content["text"] == "Compare these images"

        # Check image contents
        for i in range(1, 4):
            image_content = message.content[i]
            assert image_content["type"] == "image_url"
            assert "image_url" in image_content
            assert image_content["image_url"]["url"].startswith("data:image/jpeg;base64,")

    finally:
        # Clean up
        for tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def test_append_mixed_local_and_remote_images():
    """Test that append_image_message works with mixed local and remote images."""
    convo = MockConversation()

    # Create a temporary local image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        img = Image.new("RGB", (100, 100), color="red")
        img.save(tmp_file.name, "JPEG")
        tmp_path = tmp_file.name

    try:
        # Mock the encode_image_from_url function to avoid actual HTTP requests
        with patch("biochatter.llm_connect.conversation.encode_image_from_url") as mock_encode_url:
            mock_encode_url.return_value = "fake_base64_data"

            # Test with mixed local and remote images
            image_urls = [
                tmp_path,  # local file
                "https://example.com/image.jpg",  # remote URL
            ]

            convo.append_image_message("Compare these images", image_urls, local=False)

            # Verify the message was added correctly
            assert len(convo.messages) == 1
            message = convo.messages[0]
            assert isinstance(message, HumanMessage)
            assert isinstance(message.content, list)
            assert len(message.content) == 3  # text + 2 images

            # Verify mock was called for the remote URL
            mock_encode_url.assert_called_once_with("https://example.com/image.jpg")

    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_query_single_image():
    """Test that query method works with a single image (backward compatibility)."""
    convo = MockConversation()

    # Create a temporary image file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(tmp_file.name, "JPEG")
        tmp_path = tmp_file.name

    try:
        # Test query with single image
        result, token_usage, correction = convo.query("What do you see in this image?", image_url=tmp_path)

        # Verify the query worked
        assert result == "test response"
        assert token_usage["total_tokens"] == 100
        assert correction is None  # correct=False in MockConversation

        # Verify the message was added correctly
        assert len(convo.messages) == 1
        message = convo.messages[0]
        assert isinstance(message, HumanMessage)
        assert isinstance(message.content, list)
        assert len(message.content) == 2  # text + 1 image

        # Check content structure
        assert message.content[0]["type"] == "text"
        assert message.content[0]["text"] == "What do you see in this image?"
        assert message.content[1]["type"] == "image_url"

    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_query_multiple_images():
    """Test that query method works with multiple images."""
    convo = MockConversation()

    # Create temporary image files
    tmp_paths = []
    for i in range(2):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            colors = ["purple", "orange"]
            img = Image.new("RGB", (100, 100), color=colors[i])
            img.save(tmp_file.name, "JPEG")
            tmp_paths.append(tmp_file.name)

    try:
        # Test query with multiple images
        result, token_usage, correction = convo.query("Compare these images", image_url=tmp_paths)

        # Verify the query worked
        assert result == "test response"
        assert token_usage["total_tokens"] == 100
        assert correction is None

        # Verify the message was added correctly
        assert len(convo.messages) == 1
        message = convo.messages[0]
        assert isinstance(message, HumanMessage)
        assert isinstance(message.content, list)
        assert len(message.content) == 3  # text + 2 images

        # Check content structure
        assert message.content[0]["type"] == "text"
        assert message.content[0]["text"] == "Compare these images"

        # Check both images
        for i in range(1, 3):
            assert message.content[i]["type"] == "image_url"
            assert "image_url" in message.content[i]
            assert message.content[i]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    finally:
        # Clean up
        for tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def test_query_no_image():
    """Test that query method works without images (text only)."""
    convo = MockConversation()

    # Test query without image
    result, token_usage, correction = convo.query("Hello, how are you?")

    # Verify the query worked
    assert result == "test response"
    assert token_usage["total_tokens"] == 100
    assert correction is None

    # Verify the message was added correctly
    assert len(convo.messages) == 1
    message = convo.messages[0]
    assert isinstance(message, HumanMessage)
    assert isinstance(message.content, str)  # Text-only message has string content
    assert message.content == "Hello, how are you?"


def test_query_with_local_parameter():
    """Test that query method properly passes the local parameter to append_image_message."""
    convo = MockConversation()

    # Create a temporary image file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        img = Image.new("RGB", (100, 100), color="cyan")
        img.save(tmp_file.name, "JPEG")
        tmp_path = tmp_file.name

    try:
        # Test query with local=True (should work with local files)
        result, token_usage, correction = convo.query("Analyze this local image", image_url=tmp_path, local=True)

        # Verify the query worked
        assert result == "test response"
        assert token_usage["total_tokens"] == 100
        assert correction is None

        # Verify the message was added correctly
        assert len(convo.messages) == 1
        message = convo.messages[0]
        assert isinstance(message, HumanMessage)
        assert isinstance(message.content, list)
        assert len(message.content) == 2  # text + 1 image

        # Check that the image was encoded as base64 (indicates local=True worked)
        assert message.content[1]["type"] == "image_url"
        assert message.content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

        # Test with multiple images and local=True
        convo.messages.clear()
        result, token_usage, correction = convo.query(
            "Compare these local images",
            image_url=[tmp_path, tmp_path],  # Use same image twice for simplicity
            local=True,
        )

        # Verify multiple images work with local parameter
        assert result == "test response"
        message = convo.messages[0]
        assert len(message.content) == 3  # text + 2 images

        # Both images should be encoded as base64
        for i in [1, 2]:
            assert message.content[i]["type"] == "image_url"
            assert message.content[i]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
