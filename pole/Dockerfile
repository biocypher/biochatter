FROM docker.io/andimajore/biocyper_base:python3.10 as setup-stage

#Sets default for BIOCYPHER_CONFIG
ARG BIOCYPHER_CONFIG=config/biocypher_docker_config.yaml
ENV USED_BIOCYPHER_CONFIG=$BIOCYPHER_CONFIG

WORKDIR /usr/app/
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && poetry install
COPY . ./
RUN cp ${USED_BIOCYPHER_CONFIG} config/biocypher_config.yaml
RUN python3 create_knowledge_graph.py

FROM docker.io/neo4j:4.4-enterprise as deploy-stage
COPY --from=setup-stage /usr/app/biocypher-out/ /var/lib/neo4j/import/
COPY docker/* ./
RUN cat biocypher_entrypoint_patch.sh | cat - /startup/docker-entrypoint.sh > docker-entrypoint.sh && mv docker-entrypoint.sh /startup/ && chmod +x /startup/docker-entrypoint.sh
