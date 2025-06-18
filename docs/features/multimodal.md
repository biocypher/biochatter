# Multimodal models - Text and image

We support multimodal queries in models that offer these capabilities after the
blueprint of the OpenAI API. We can either add an image-containing message to
the conversation using the `append_image_message` method, or we can pass an
image URL directly to the `query` method:

```python
# Either: Append image message
conversation.append_image_message(
    message="Here is an attached image",
    image_url="https://example.com/image.jpg"
)

# Or: Query with image included
msg, token_usage, correction = conversation.query(
    "What's in this image?",
    image_url="https://example.com/image.jpg"
)
```

### Using local images

Following the recommendations by OpenAI, we can pass local images as
base64-encoded strings. We allow this by setting the `local` flag to `True` in
the `append_image_message` method:

```python
conversation.append_image_message(
    message="Here is an attached image",
    image_url="my/local/image.jpg",
    local=True
)
```

We also support the use of local images in the `query` method by detecting the
netloc of the image URL. If the netloc is empty, we assume that the image is
local and read it as a base64-encoded string:

```python
msg, token_usage, correction = conversation.query(
    "What's in this image?",
    image_url="my/local/image.jpg"
)
```

### Parsing PDF as images

Recent multimodal models achieve strong performance on image processing tasks.
To leverage this, we provide the `pdf_pages_to_images` utility, which converts a
PDF into a list of page images. These images can be passed to the `query`
method to enrich the context of your queries.

Since PDF parsing can be noisy, especially when documents contain many embedded
images, this option allows you to pass a list of pre-extracted images directly
to the `query` method, enabling a fast document question-answering.

```python
from biochatter.utils import pdf_pages_to_images
from biochatter.llm_connect import LangChainConversation

convo = LangChainConversation(
    model_name="gemini-2.5-flash-05-20-preview",
    model_provider="google_genai",
    prompts={},
)

convo.set_api_key()

pdf_path = "/path/to/your/pdf.pdf"
output_folder = "/path/to/output/folder"
images = pdf_pages_to_images(pdf_path, output_folder)

convo.query(
    "Summarize the content of this document",
    image_url=images,
)
```

### Open-source multimodal models

While OpenAI models work seamlessly, open-source multimodal models can be buggy
or incompatible with certain hardware. We have experienced mixed success with
open models and, while they are technically supported by BioChatter, their
outputs currently may be unreliable.
