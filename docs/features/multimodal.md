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

### Open-source multimodal models

While OpenAI models work seamlessly, open-source multimodal models can be buggy
or incompatible with certain hardware. We have experienced mixed success with
open models and, while they are technically supported by BioChatter, their
outputs currently may be unreliable.
