# Basic Usage: Chat

BioChatter provides access to chat functionality via the `Conversation` class,
which is implemented in several child classes (in the `llm_connect.py` module)
to account for differences in APIs of the LLMs.

## Setting up the conversation

To start a conversation, we can initialise the Conversation class (here
exemplified by GPT):

```python
from biochatter.llm_connect import GptConversation

conversation = GptConversation(
    model_name="gpt-3.5-turbo",
    prompts={},
)
```

It is possible to supply a dictionary of prompts to the conversation from the
outset, which is formatted in a way to correspond to the different roles of the
conversation, i.e., primary and correcting models. Prompts with the
`primary_model_prompts` key will be appended to the System Messages of the
primary model, and `correcting_agent_prompts` will be appended to the System
Messages of the correction model at setup. If we pass a dictionary without these
keys (or an empty one), there will be no system messages appended to the models.
They can however be introduced later by using the following method:

```python
conversation.append_system_message("System Message")
```

Similarly, the user queries (`HumanMessage`) are passed to the conversation
using `conversation.append_user_message("User Message")`. For purposes of
keeping track of the conversation history, we can also append the model's
responses as `AIMessage` using `conversation.append_ai_message`.

## Querying the model

After setting up the conversation in this way, for instance by establishing a
flattery component (e.g. 'You are an assistant to a researcher ...'), the model
can be queried using the `query` function.

```python
msg, token_usage, correction = conversation.query('Question here')
```

Note that a query will automatically append a user message to the message
history, so there is no need to call `append_user_message()` again. The query
function returns the actual answer of the model (`msg`), the token usage
statistics reported by the API (`token_usage`), and an optional `correction`
that contains the opinion of the corrective agent.

## Using OpenAI models

Using an OpenAI model via the API is generally the easiest way to get started,
but requires the provision of an API key to the OpenAI API. To do this, you can
designate the `OPENAI_API_KEY` variable in your environment directly (`export
OPENAI_API_KEY=sk-...`) by adding it to your shell configuration (e.g., the
`zshrc`).

## Multimodal models - Text and image

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