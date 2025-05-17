If you are asking yourself "what is this even used for?", this page is for you.

Large language models (LLMs) have inspired many creative applications in the
recent years, and our aim is to channel these applications into a framework
that offers more robust and scalable solutions for the biomedical domain. We see
BioChatter as a developer toolkit that can be used to create a range of custom
applications for specific user bases. As such, our main target audience are
developers in the biomedical domain.

## Chatbots

Naturally, the most attractive general use of LLMs is to have a language
interface to some functionality that previously was not accessible through such
simple means. Depending on the requirements of the application, this can be
either trivial ("Create a paragraph from these bullet points.") or extremely
complex ("What is the driver of my patient's cancer?"). Biomedical use cases
tend to be biased towards the latter, often necessitating the integration of
various sources of knowledge and quality assurance of the conversational system.
As a result, BioChatter is not designed to function as a sole chatbot without
integration of other data or knowledge; this functionality is provided by
numerous companies.

Instead, we at least include specific prompts that tune the LLM towards the
desired application and user base, which can be configured by the developer
implementing the application. More powerful applications are the focus however,
including [tool use](../features/api.md) (external software parameterisation),
[knowledge graph](../vignettes/kg.md) and [vector database](../vignettes/rag.md)
integration, [multimodal inputs](../features/multimodal.md), and modular
[agentic workflows](../features/reflexion-agent.md).

## Graphical user interfaces

Increasing accessibility to previously inaccessible functionality naturally
implies that the developer needs to think about how to make the functionality
available to the (by definition non-technical) user. As a result, we include
suggestions for diverse graphical user interface (GUI) frameworks in the
ecosystem. Specifically, we provide instructive examples for relatively simple
prototyping frameworks based on [Streamlit](https://streamlit.io/), which we
brand as [BioChatter Light](https://light.biochatter.org) and which can be used
for rapid development of a range of user interfaces:

- [simple configuration](../vignettes/custom-bclight-simple.md)

- [advanced configuration](../vignettes/custom-bclight-advanced.md)

- [real-world use case](../vignettes/custom-decider-use-case.md)

For more complex applications, we also provide a modular system based on current
web technologies ([FastAPI](https://fastapi.tiangolo.com/) and
[Next.js](https://nextjs.org/)), which we brand as [BioChatter
Next](https://next.biochatter.org). The web application is driven by a RESTful
API, which we implement in [BioChatter
Server](https://github.com/biocypher/biochatter-server). An application example
can be seen in the [real-world use
case](../vignettes/custom-decider-use-case.md).
