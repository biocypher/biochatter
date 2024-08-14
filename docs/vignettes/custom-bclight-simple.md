# Custom BioChatter Light: Simple Use Case

For prototyping your own text2cypher workflow, it can be useful to have a simple
interface for troubleshooting your queries. This vignette demonstrates how to
customise the pre-built BioChatter Light Docker compose to turn off all tabs
except the Knowledge Graph (KG) tab. This allows the deployment of an integrated
KG build, deployment, and web app for LLM-based question answering.

## Build your KG

First, build your KG. For this example, we use the [BioCypher Pole
KG](https://github.com/biocypher/pole) as a demo KG. The KG is based on an
open-source dataset of crime statistics in Manchester. The schema of the demo KG
is described in the [Knowledge Graph RAG](kg.md) vignette. For building
your own KG, refer to the [BioCypher documentation](https://biocypher.org).

This KG is built, imported, and deployed in the first three stages of the
`docker-compose.yml` file.

## Configure the BioChatter Light Docker container

We provide a simple way to customise the BioChatter Light Docker container to
show only select components. We can provide these settings via environment
variables, so in the case of running from `docker-compose.yml`, we can set these
in the `environment` section of the `app` service.

```yaml
services:
  ## ... build, import, and deploy the KG ...
  app:
    image: biocypher/biochatter-light:0.6.10
    container_name: app
    ports:
      - "8501:8501"
    networks:
      - biochatter
    depends_on:
      import:
        condition: service_completed_successfully
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DOCKER_COMPOSE=true
      - CHAT_TAB=false
      - PROMPT_ENGINEERING_TAB=false
      - RAG_TAB=false
      - CORRECTING_AGENT_TAB=false
      - KNOWLEDGE_GRAPH_TAB=true
```

In this example, we provide our OpenAI API key and set the `DOCKER_COMPOSE` flag
to `true`, which tells BioChatter Light to connect to the KG on the Docker
network, which uses the service name as the hostname, so in this case, `deploy`
instead of the default `localhost`.

We then turn off all default tabs (chatting, prompt engineering, RAG, and the
correcting agent) and turn on the KG tab. Running the docker compose with these
settings will build and deploy the KG and the BioChatter Light web app with only
the KG tab enabled.

```bash
git clone https://github.com/biocypher/pole
cd pole
docker-compose up -d
```

We are constantly expanding our [repertoire of BioChatter Light
tabs](https://github.com/biocypher/biochatter-light?tab=readme-ov-file#tab-selection),
so check back for more options in the future. Creating your own tabs is also
accessible via our modular architecture and the simple
[Streamlit](https://streamlit.io) framework for UI design. Check the [advanced
vignette](custom-bclight-advanced.md) and the codebase of [BioChatter
Light](https://github.com/biocypher/biochatter-light) for more information.
