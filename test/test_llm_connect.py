# ruff: noqa: S101  # Use of assert detected
# ruff: noqa: ANN201  # No docstring in public function
# ruff: noqa: D103  # Missing docstring in public function
# ruff: noqa: D100  # Missing docstring in public module

import os
from unittest.mock import MagicMock, Mock, mock_open, patch

import openai
import pytest
from openai._exceptions import NotFoundError
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
    AIMessage,
    AnthropicConversation,
    AzureGptConversation,
    GptConversation,
    HumanMessage,
    OllamaConversation,
    SystemMessage,
    WasmConversation,
    XinferenceConversation,
)


@pytest.fixture(scope="module", autouse=True)
def manage_test_context():
    import openai

    base_url = openai.base_url
    api_type = openai.api_type
    api_version = openai.api_version
    api_key = openai.api_key
    organization = openai.organization
    proxy = getattr(openai, "proxy", None)
    yield True

    openai.base_url = base_url
    openai.api_type = api_type
    openai.api_version = api_version
    openai.api_key = api_key
    openai.organization = organization
    if proxy is not None:
        openai.proxy = proxy
    elif hasattr(openai, "proxy"):
        delattr(openai, "proxy")


def test_empty_messages():
    convo = GptConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )
    assert convo.get_msg_json() == "[]"


def test_single_message():
    convo = GptConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )
    convo.messages.append(SystemMessage(content="Hello, world!"))
    assert convo.get_msg_json() == '[{"system": "Hello, world!"}]'


def test_multiple_messages():
    convo = GptConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )
    convo.messages.append(SystemMessage(content="Hello, world!"))
    convo.messages.append(HumanMessage(content="How are you?"))
    convo.messages.append(AIMessage(content="I'm doing well, thanks!"))
    assert convo.get_msg_json() == (
        '[{"system": "Hello, world!"}, {"user": "How are you?"}, {"ai": "I\'m doing well, thanks!"}]'
    )


def test_unknown_message_type():
    convo = GptConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )
    convo.messages.append(None)
    with pytest.raises(ValueError):
        convo.get_msg_json()


@patch("biochatter.llm_connect.openai.OpenAI")
def test_openai_catches_authentication_error(mock_openai):
    mock_openai.return_value.models.list.side_effect = openai._exceptions.AuthenticationError(
        (
            "Incorrect API key provided: fake_key. You can find your API key"
            " at https://platform.openai.com/account/api-keys."
        ),
        response=Mock(),
        body=None,
    )
    convo = GptConversation(
        model_name="gpt-3.5-turbo",
        prompts={},
        split_correction=False,
    )

    success = convo.set_api_key(
        api_key="fake_key",
        user="test_user",
    )

    assert not success


@patch("biochatter.llm_connect.AzureChatOpenAI")
def test_azure_raises_request_error(mock_azure_chat):
    mock_azure_chat.side_effect = NotFoundError(
        message="Resource not found",
        response=Mock(),
        body=None,
    )

    convo = AzureGptConversation(
        model_name="gpt-35-turbo",
        deployment_name="test_deployment",
        prompts={},
        split_correction=False,
        version="2023-03-15-preview",
        base_url="https://api.openai.com",
    )

    with pytest.raises(NotFoundError):
        convo.set_api_key("fake_key")


@patch("biochatter.llm_connect.AzureChatOpenAI")
def test_azure(mock_azure_chat):
    """Test OpenAI Azure endpoint functionality.

    Azure connectivity is enabled by setting the corresponding environment
    variables.
    """
    convo = AzureGptConversation(
        model_name=os.getenv("AZURE_TEST_OPENAI_MODEL_NAME"),
        deployment_name=os.getenv("AZURE_TEST_OPENAI_DEPLOYMENT_NAME"),
        prompts={},
        split_correction=False,
        version=os.getenv("AZURE_TEST_OPENAI_API_VERSION"),
        base_url=os.getenv("AZURE_TEST_OPENAI_API_BASE"),
    )

    mock_azure_chat.return_value = Mock()

    assert convo.set_api_key(os.getenv("AZURE_TEST_OPENAI_API_KEY"))


xinference_models = {
    "48c76b62-904c-11ee-a3d2-0242acac0302": {
        "model_type": "embedding",
        "address": "",
        "accelerators": ["0"],
        "model_name": "gte-large",
        "dimensions": 1024,
        "max_tokens": 512,
        "language": ["en"],
        "model_revision": "",
    },
    "a823319a-88bd-11ee-8c78-0242acac0302": {
        "model_type": "LLM",
        "address": "0.0.0.0:46237",
        "accelerators": ["0"],
        "model_name": "llama2-13b-chat-hf",
        "model_lang": ["en"],
        "model_ability": ["embed", "generate", "chat"],
        "model_format": "pytorch",
        "context_length": 4096,
    },
}


@pytest.mark.skip(reason="Live test for development purposes")
def test_anthropic():
    conv = AnthropicConversation(
        model_name="claude-3-5-sonnet-20240620",
        prompts={},
        split_correction=False,
    )
    assert conv.set_api_key(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        user="test_user",
    )


def test_xinference_init():
    """Test generic LLM connectivity via the Xinference client. Currently depends
    on a test server.
    """
    base_url = os.getenv("XINFERENCE_BASE_URL", "http://localhost:9997")
    with patch("xinference.client.Client") as mock_client:
        mock_client.return_value.list_models.return_value = xinference_models
        convo = XinferenceConversation(
            base_url=base_url,
            prompts={},
            split_correction=False,
        )
        assert convo.set_api_key()


def test_xinference_chatting():
    base_url = os.getenv("XINFERENCE_BASE_URL", "http://localhost:9997")
    with patch("xinference.client.Client") as mock_client:
        response = {
            "id": "1",
            "object": "chat.completion",
            "created": 123,
            "model": "foo",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": " Hello there, can you sing me a song?",
                    },
                    "finish_reason": "stop",
                },
            ],
            "usage": {
                "prompt_tokens": 93,
                "completion_tokens": 54,
                "total_tokens": 147,
            },
        }
        mock_client.return_value.list_models.return_value = xinference_models
        mock_client.return_value.get_model.return_value.chat.return_value = response
        convo = XinferenceConversation(
            base_url=base_url,
            prompts={},
            correct=False,
        )
        (msg, token_usage, correction) = convo.query("Hello, world!")
        assert token_usage["completion_tokens"] > 0


def test_ollama_chatting():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    with patch("biochatter.llm_connect.ChatOllama") as mock_model:
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


@pytest.fixture
def xinference_conversation():
    with patch("xinference.client.Client") as mock_client:
        # Mock the authentication check
        mock_client.return_value._check_cluster_authenticated.return_value = None
        mock_client.return_value.list_models.return_value = xinference_models
        mock_client.return_value.get_model.return_value.chat.return_value = (
            {"choices": [{"message": {"content": "Human message"}}]},
            {"completion_tokens": 0},
        )
        conversation = XinferenceConversation(
            base_url="http://localhost:9997",
            prompts={},
            correct=False,
        )
        return conversation


def test_single_system_message_before_human(xinference_conversation: XinferenceConversation):
    xinference_conversation.messages = [
        SystemMessage(content="System message"),
        HumanMessage(content="Human message"),
    ]
    history = xinference_conversation._create_history()
    assert history.pop() == {
        "role": "user",
        "content": "System message\nHuman message",
    }


def test_multiple_system_messages_before_human(xinference_conversation: XinferenceConversation):
    xinference_conversation.messages = [
        SystemMessage(content="System message 1"),
        SystemMessage(content="System message 2"),
        HumanMessage(content="Human message"),
    ]
    history = xinference_conversation._create_history()
    assert history.pop() == {
        "role": "user",
        "content": "System message 1\nSystem message 2\nHuman message",
    }


def test_multiple_messages_including_ai_before_system_and_human(
    xinference_conversation: XinferenceConversation,
):
    xinference_conversation.messages = [
        HumanMessage(content="Human message history"),
        AIMessage(content="AI message"),
        SystemMessage(content="System message"),
        HumanMessage(content="Human message"),
    ]
    history = xinference_conversation._create_history()
    assert history.pop() == {
        "role": "user",
        "content": "System message\nHuman message",
    }


def test_multiple_cycles_of_ai_and_human(xinference_conversation):
    xinference_conversation.messages = [
        HumanMessage(content="Human message history"),
        AIMessage(content="AI message"),
        HumanMessage(content="Human message"),
        AIMessage(content="AI message"),
        HumanMessage(content="Human message"),
        AIMessage(content="AI message"),
        SystemMessage(content="System message"),
        HumanMessage(content="Human message"),
    ]
    history = xinference_conversation._create_history()
    assert len(history) == 3
    assert history.pop() == {
        "role": "user",
        "content": "System message\nHuman message",
    }


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
    with patch("biochatter.llm_connect.urllib.request.urlopen") as mock_urlopen:
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


@pytest.mark.skip(reason="Live test for development purposes")
def test_append_local_image_gpt():
    convo = GptConversation(
        model_name="gpt-4o",
        prompts={},
        correct=False,
        split_correction=False,
    )
    convo.set_api_key(api_key=os.getenv("OPENAI_API_KEY"), user="test_user")

    convo.append_system_message(
        "You are an editorial assistant to a journal in biomedical science.",
    )

    convo.append_image_message(
        message=(
            "This text describes the attached image: "
            "Live confocal imaging of liver stage P. berghei expressing UIS4-mCherry and cytoplasmic GFP reveals different morphologies of the LS-TVN: elongated membrane clusters (left), vesicles in the host cell cytoplasm (center), and a thin tubule protruding from the PVM (right). Live imaging was performed 20?h after infection of hepatoma cells. Features are marked with white arrowheads."
        ),
        image_url="test/figure_panel.jpg",
        local=True,
    )

    result, _, _ = convo.query("Is the description accurate?")
    assert "yes" in result.lower()


@pytest.mark.skip(reason="Live test for development purposes")
def test_local_image_query_gpt():
    convo = GptConversation(
        model_name="gpt-4o",
        prompts={},
        correct=False,
        split_correction=False,
    )
    convo.set_api_key(api_key=os.getenv("OPENAI_API_KEY"), user="test_user")

    convo.append_system_message(
        "You are an editorial assistant to a journal in biomedical science.",
    )

    result, _, _ = convo.query(
        "Does this text describe the attached image: Live confocal imaging of liver stage P. berghei expressing UIS4-mCherry and cytoplasmic GFP reveals different morphologies of the LS-TVN: elongated membrane clusters (left), vesicles in the host cell cytoplasm (center), and a thin tubule protruding from the PVM (right). Live imaging was performed 20?h after infection of hepatoma cells. Features are marked with white arrowheads.",
        image_url="test/figure_panel.jpg",
    )
    assert "yes" in result.lower()


@pytest.mark.skip(reason="Live test for development purposes")
def test_append_online_image_gpt():
    convo = GptConversation(
        model_name="gpt-4o",
        prompts={},
        correct=False,
        split_correction=False,
    )
    convo.set_api_key(api_key=os.getenv("OPENAI_API_KEY"), user="test_user")

    convo.append_image_message(
        "This is a picture from the internet.",
        image_url="https://upload.wikimedia.org/wikipedia/commons/8/8f/The-Transformer-model-architecture.png",
    )

    result, _, _ = convo.query("What does this picture show?")
    assert "transformer" in result.lower()


@pytest.mark.skip(reason="Live test for development purposes")
def test_online_image_query_gpt():
    convo = GptConversation(
        model_name="gpt-4o",
        prompts={},
        correct=False,
        split_correction=False,
    )
    convo.set_api_key(api_key=os.getenv("OPENAI_API_KEY"), user="test_user")

    result, _, _ = convo.query(
        "What does this picture show?",
        image_url="https://upload.wikimedia.org/wikipedia/commons/8/8f/The-Transformer-model-architecture.png",
    )
    assert "transformer" in result.lower()


@pytest.mark.skip(reason="Live test for development purposes")
def test_local_image_query_xinference():
    url = "http://localhost:9997"
    convo = XinferenceConversation(
        base_url=url,
        prompts={},
        correct=False,
    )
    assert convo.set_api_key()

    result, _, _ = convo.query(
        "Does this text describe the attached image: Live confocal imaging of liver stage P. berghei expressing UIS4-mCherry and cytoplasmic GFP reveals different morphologies of the LS-TVN: elongated membrane clusters (left), vesicles in the host cell cytoplasm (center), and a thin tubule protruding from the PVM (right). Live imaging was performed 20?h after infection of hepatoma cells. Features are marked with white arrowheads.",
        image_url="test/figure_panel.jpg",
    )
    assert isinstance(result, str)

