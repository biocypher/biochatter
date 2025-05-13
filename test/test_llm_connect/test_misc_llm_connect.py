"""Tests for the miscellaneous LLM connect module."""

import os
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

from langchain_core.messages import AIMessage


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
        query_result = convo.query("Hello, world!")
        assert query_result.token_usage > 0


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
    query_result = wasm_convo.query(test_query)
    assert query_result.response == test_query  # assuming the messages list is initially empty

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
