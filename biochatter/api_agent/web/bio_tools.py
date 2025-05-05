"""Module for interacting with the bio.tools API."""

import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING

import requests
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

if TYPE_CHECKING:
    from biochatter.llm_connect import Conversation

from biochatter.api_agent.base.agent_abc import (
    BaseFetcher,
    BaseInterpreter,
    BaseQueryBuilder,
)

BIOTOOLS_QUERY_PROMPT = """
You are a world class algorithm for creating queries in structured formats. Your task is to use the web API of bio.tools
to answer questions about bioinformatics tools and their properties.

You have to extract the appropriate information out of the examples:
1. To list information about the tools, use the endpoint https://bio.tools/api/t/ with parameters like name,
description, homepage, etc.

Use these formats to generate queries based on the question provided. Below is more information about the bio.tools API:

Base URL

https://bio.tools/api/

Endpoints and Parameters

1. List tools

GET /t/

==================  ============================================================================================
Parameter           Search behaviour
==================  ============================================================================================
biotoolsID          Search for bio.tools tool ID (usually quoted - to get exact match)

                    `biotoolsID="signalp" <https://bio.tools/api/t/?biotoolsID="signalp">`_

name                Search for tool name (quoted as needed)

                    `name=signalp <https://bio.tools/api/t/?name=signalp>`_
homepage            Exact search for tool homepage URL (**must** be quoted)

                    `homepage="http://cbs.dtu.dk/services/SignalP/" <https://bio.tools/api/t/?homepage="http://cbs.dtu.dk/services/SignalP/">`_
description         Search over tool description (quoted as needed)

                    `description="peptide cleavage" <https://bio.tools/api/t/?description="peptide%20cleavage">`_
version             Exact search for tool version (**must** be quoted)

                    `version="4.1" <https://bio.tools/api/t/?version="4.1">`_
topic               Search for EDAM Topic (term) (quoted as needed)

                    `topic="Proteomics" <https://bio.tools/api/t/?topic="Proteomics">`_

topicID             Exact search for EDAM Topic (URI): **must** be quoted

                    `topicID="topic_3510" <https://bio.tools/api/t/?topicID="topic_3510">`_
function            Fuzzy search over function (input, operation, output, note and command)

                    `function="Sequence analysis" <https://bio.tools/api/t/?function="Sequence%20analysis">`_
operation           Fuzzy search for EDAM Operation (term) (quoted as needed)

                    `operation="Sequence analysis" <https://bio.tools/api/t/?operation="Sequence%20analysis">`_
operationID         Exact search for EDAM Operation (ID) (**must** be quoted)

                    `operationID="operation_2403" <https://bio.tools/api/t/?operationID="operation_2403">`_
dataType            Fuzzy search over input and output for EDAM Data (term) (quoted as needed)

                    `dataType="Protein sequence" <https://bio.tools/api/t/?dataType="Protein%20sequence">`_
dataTypeID          Exact search over input and output for EDAM Data (ID) (**must** be quoted)

                    `dataTypeID="data_2976" <https://bio.tools/api/t/?dataTypeID="data_2976">`_
dataFormat          Fuzzy search over input and output for EDAM Format (term) (quoted as needed)

                    `dataFormat="FASTA" <https://bio.tools/api/t/?dataFormat="FASTA">`_
dataFormatID        Exact search over input and output for EDAM Format (ID) (**must** be quoted)

                    `dataFormatID="format_1929" <https://bio.tools/api/t/?dataFormatID="format_1929">`_
input               Fuzzy search over input for EDAM Data and Format (term) (quoted as needed)

                    `input="Protein sequence" <https://bio.tools/api/t/?input="Protein%20sequence">`_
inputID             Exact search over input for EDAM Data and Format (ID) (**must** be quoted)

                    `inputID="data_2976" <https://bio.tools/api/t/?inputID="data_2976">`_
inputDataType       Fuzzy search over input for EDAM Data (term) (quoted as needed)

                    `inputDataType="Protein sequence" <https://bio.tools/api/t/?inputDataType="Protein%20sequence">`_
inputDataTypeID     Exact search over input for EDAM Data (ID) (**must** be quoted)

                    `inputDataTypeID="data_2976" <https://bio.tools/api/t/?inputDataTypeID="data_2976">`_
inputDataFormat     Fuzzy search over input for EDAM Format (term) (quoted as needed)

                    `inputDataFormat="FASTA" <https://bio.tools/api/t/?inputDataFormat="FASTA">`_
inputDataFormatID   Exact search over input for EDAM Format (ID) (**must** be quoted)

                    `inputDataFormatID="format_1929" <https://bio.tools/api/t/?inputDataFormatID="format_1929">`_
output              Fuzzy search over output for EDAM Data and Format (term) (quoted as needed)

                    `output="Sequence alignment" <https://bio.tools/api/t/?output="Sequence%20alignment">`_
outputID            Exact search over output for EDAM Data and Format (ID) (**must** be quoted)

                    `outputID="data_0863" <https://bio.tools/api/t/?outputID="data_0863">`_
outputDataType      Fuzzy search over output for EDAM Data (term) (quoted as needed)

                    `outputDataType="Sequence alignment" <https://bio.tools/api/t/?outputDataType="Sequence%20alignment">`_
outputDataTypeID    Exact search over output for EDAM Data (ID) (**must** be quoted)

                    `outputDataTypeID="data_0863" <https://bio.tools/api/t/?outputDataTypeID="data_0863">`_
outputDataFormat    Fuzzy search over output for EDAM Format (term) (quoted as needed)

                    `outputDataFormat="ClustalW format" <https://bio.tools/api/t/?outputDataFormat="ClustalW%20format">`_
outputDataFormatID  Exact search over output for EDAM Format (ID) (**must** be quoted)

                    `outputDataFormatID="format_1982" <https://bio.tools/api/t/?outputDataFormatID="format_1982">`_
toolType            Exact search for tool type

                    `toolType="Command-line tool" <https://bio.tools/api/t/?toolType="Command-line%20tool">`_
collectionID        Exact search for tool collection (normally quoted)

                    `collectionID="Rare Disease" <https://bio.tools/api/t/?collectionID="Rare%20Disease">`_
maturity            Exact search for tool maturity

                    `maturity=Mature <https://bio.tools/api/t/?maturity=Mature>`_
operatingSystem     Exact search for tool operating system

                    `operatingSystem=Linux <https://bio.tools/api/t/?operatingSystem=Linux>`_
language            Exact search for programming language

                    `language=Java <https://bio.tools/api/t/?language=Java>`_
cost                Exact search for cost

                    `cost="Free of charge" <https://bio.tools/api/t/?cost="Free%20of%20charge">`_
license             Exact search for software or data usage license (quoted as needed)

                    `license="GPL-3.0" <https://bio.tools/api/t/?license="GPL-3.0">`_
accessibility       Exact search for tool accessibility

                    `accessibility="Open access" <https://bio.tools/api/t/?accessibility="Open%20access">`_
credit              Fuzzy search over credit (name, email, URL, ORCID iD, type of entity, type of role and note)

                    `credit="Henrik Nielsen" <https://bio.tools/api/t/?credit="Henrik%20Nielsen">`_
creditName          Exact search for name of credited entity

                    `creditName="Henrik Nielsen" <https://bio.tools/api/t/?creditName="Henrik%20Nielsen">`_
creditTypeRole      Exact search for role of credited entity

                    `creditTypeRole=Developer <https://bio.tools/api/t/?creditTypeRole=Developer>`_
creditTypeEntity    Exact search for type of credited entity

                    `creditTypeEntity="Funding agency" <https://bio.tools/api/t/?creditTypeEntity="Funding%20agency">`_
creditOrcidID       Exact search for ORCID iD of credited entity (**must** be quoted)

                    `creditOrcidID="0000-0001-5121-2036" <https://bio.tools/api/t/?creditOrcidID="0000-0001-5121-2036">`_
publication         Fuzzy search over publication (DOI, PMID, PMCID, publication type and tool version) (quoted as needed)

                    `publication=10.12688/f1000research.12974.1 <https://bio.tools/api/t/?publication=10.12688/f1000research.12974.1>`_
publicationID       Exact search for publication ID (DOI, PMID or PMCID) (**must** be quoted)

                    `publicationID="10.12688/f1000research.12974.1" <https://bio.tools/api/t/?publicationID="10.12688/f1000research.12974.1">`_
publicationType     Exact search for publication type

                    `publicationType=Primary <https://bio.tools/api/t/?publicationType=Primary>`_
publicationVersion  Exact search for tool version associated with a publication (**must** be quoted)

                    `publicationVersion="1.0" <https://bio.tools/api/t/?publicationVersion="1.0">`_
link                Fuzzy search over general link (URL, type and note) (quote as needed)

                    `link="Issue tracker" <https://bio.tools/api/t/?link="Issue%20tracker">`_
linkType            Exact search for type of information found at a link

                    `linkType="Issue tracker" <https://bio.tools/api/t/?linkType="Issue tracker">`_
documentation       Fuzzy search over documentation link (URL, type and note) (quote as needed)

                    `documentation=Manual <https://bio.tools/api/t/?documentation="User manual">`_
documentationType   Exact search for type of documentation

                    `documentationType=Manual <https://bio.tools/api/t/?documentationType="User manual">`_
download            Fuzzy search over download link (URL, type, version and note) (quote as needed)

                    `download=Binaries <https://bio.tools/api/t/?download=Binaries>`_
downloadType        Exact search for type of download

                    `downloadType=Binaries <https://bio.tools/api/t/?downloadType=Binaries>`_
downloadVersion     Exact search for tool version associated with a download (**must** be quoted)

                    `downloadVersion="1.0" <https://bio.tools/api/t/?downloadVersion="1.0">`_
otherID             Fuzzy search over alternate tool IDs (ID value, type of ID and version)

                    `otherID="rrid:SCR_015644" <https://bio.tools/api/t/?otherID="rrid:SCR_015644">`_

otherIDValue        Exact search for value of alternate tool ID (**must** be quoted)

                    `otherIDValue="rrid:SCR_015644" <https://bio.tools/api/t/?otherIDValue="rrid:SCR_015644">`_
otherIDType         Exact search for type of alternate tool ID

                    `otherIDType=RRID <https://bio.tools/api/t/?otherIDType=RRID>`_
otherIDVersion      Exact search for tool version associated with an alternate ID (**must** be quoted)

                    `otherIDVersion="1.0" <https://bio.tools/api/t/?otherIDVersion="1.0">`_
==================  ============================================================================================


The parameters are (currently) case-sensitive, e.g. you must use &biotoolsID= and not &biotoolsid

Values of the following parameters must be given in quotes to get sensible (or any) results:
homepage
version
topicID
operationID
dataTypeID
dataFormatID
inputID
inputDataTypeID
inputDataFormatID
outputID
outputDataTypeID
outputDataFormatID
creditOrcidID
publicationID
publicationVersion
downloadVersion
otherIDValue
otherIDVersion
e.g.
https://bio.tools/api/tool?topicID=”topic_3510”
Values of other parameters can be quoted or unquoted:
Unquoted values invoke a fuzzy word search: it will search for fuzzy matches of words in the search phrase, to the
target field
Quoted values invoke an exact phrase search; it will search for an exact match of the full-length of the search phrase,
to the target field (matches to target substrings are allowed)
e.g.
https://bio.tools/api/tool?biotoolsID=”blast” returns the tool with biotoolsID of “blast” (the “canonical” blast)
https://bio.tools/api/tool?biotoolsID=blast returns all tools with “blast” in their biotoolsID (all blast flavours)
"""


BIOTOOLS_SUMMARY_PROMPT = """
You have to answer this question in a clear and concise manner: {question} Be factual!\n\
You are a world leading bioinformatician who knows everything about bio.tools packages.\n\
Do not make up information, only use the provided information and mention how relevant the found information is based on
your knowledge about bio.tools.\n\
Here is the information relevant to the question found on the bio.tools web API:\n\
{context}
"""


class BioToolsQueryParameters(BaseModel):
    """Parameters for querying the bio.tools API."""

    base_url: str = Field(
        default="https://bio.tools/api/",
        description="Base URL for the BioTools API.",
    )
    endpoint: str = Field(
        ...,
        description="Specific API endpoint to hit. Example: 't/' for listing tools.",
    )
    biotoolsID: str | None = Field(  # noqa: N815
        None,
        description="Search for bio.tools tool ID (usually quoted - to get exact match)",
    )
    name: str | None = Field(
        None,
        description="Search for tool name (quoted as needed: quoted for exact match, unquoted for fuzzy search)",
    )
    homepage: str | None = Field(
        None,
        description="Exact search for tool homepage URL (**must** be quoted)",
    )
    description: str | None = Field(
        None,
        description="Search over tool description (quoted as needed)",
    )
    version: str | None = Field(
        None,
        description="Exact search for tool version (**must** be quoted)",
    )
    topic: str | None = Field(
        None,
        description="Search for EDAM Topic (term) (quoted as needed)",
    )
    topicID: str | None = Field(  # noqa: N815
        None,
        description="Exact search for EDAM Topic (URI): **must** be quoted",
    )
    function: str | None = Field(
        None,
        description="Fuzzy search over function (input, operation, output, note and command)",
    )
    operation: str | None = Field(
        None,
        description="Fuzzy search for EDAM Operation (term) (quoted as needed)",
    )
    operationID: str | None = Field(  # noqa: N815
        None,
        description="Exact search for EDAM Operation (ID) (**must** be quoted)",
    )
    dataType: str | None = Field(  # noqa: N815
        None,
        description="Fuzzy search over input and output for EDAM Data (term) (quoted as needed)",
    )
    dataTypeID: str | None = Field(  # noqa: N815
        None,
        description="Exact search over input and output for EDAM Data (ID) (**must** be quoted)",
    )
    dataFormat: str | None = Field(  # noqa: N815
        None,
        description="Fuzzy search over input and output for EDAM Format (term) (quoted as needed)",
    )
    dataFormatID: str | None = Field(  # noqa: N815
        None,
        description="Exact search over input and output for EDAM Format (ID) (**must** be quoted)",
    )
    input: str | None = Field(
        None,
        description="Fuzzy search over input for EDAM Data and Format (term) (quoted as needed)",
    )
    inputID: str | None = Field(  # noqa: N815
        None,
        description="Exact search over input for EDAM Data and Format (ID) (**must** be quoted)",
    )
    inputDataType: str | None = Field(  # noqa: N815
        None,
        description="Fuzzy search over input for EDAM Data (term) (quoted as needed)",
    )
    inputDataTypeID: str | None = Field(  # noqa: N815
        None,
        description="Exact search over input for EDAM Data (ID) (**must** be quoted)",
    )
    inputDataFormat: str | None = Field(  # noqa: N815
        None,
        description="Fuzzy search over input for EDAM Format (term) (quoted as needed)",
    )
    inputDataFormatID: str | None = Field(  # noqa: N815
        None,
        description="Exact search over input for EDAM Format (ID) (**must** be quoted)",
    )
    output: str | None = Field(
        None,
        description="Fuzzy search over output for EDAM Data and Format (term) (quoted as needed)",
    )
    outputID: str | None = Field(  # noqa: N815
        None,
        description="Exact search over output for EDAM Data and Format (ID) (**must** be quoted)",
    )
    outputDataType: str | None = Field(  # noqa: N815
        None,
        description="Fuzzy search over output for EDAM Data (term) (quoted as needed)",
    )
    outputDataTypeID: str | None = Field(  # noqa: N815
        None,
        description="Exact search over output for EDAM Data (ID) (**must** be quoted)",
    )
    outputDataFormat: str | None = Field(  # noqa: N815
        None,
        description="Fuzzy search over output for EDAM Format (term) (quoted as needed)",
    )
    outputDataFormatID: str | None = Field(  # noqa: N815
        None,
        description="Exact search over output for EDAM Format (ID) (**must** be quoted)",
    )
    toolType: str | None = Field(  # noqa: N815
        None,
        description="Exact search for tool type",
    )
    collectionID: str | None = Field(  # noqa: N815
        None,
        description="Exact search for tool collection (normally quoted)",
    )
    maturity: str | None = Field(
        None,
        description="Exact search for tool maturity",
    )
    operatingSystem: str | None = Field(  # noqa: N815
        None,
        description="Exact search for tool operating system",
    )
    language: str | None = Field(
        None,
        description="Exact search for programming language",
    )
    cost: str | None = Field(
        None,
        description="Exact search for cost",
    )
    license: str | None = Field(
        None,
        description="Exact search for software or data usage license (quoted as needed)",
    )
    accessibility: str | None = Field(
        None,
        description="Exact search for tool accessibility",
    )
    credit: str | None = Field(
        None,
        description="Fuzzy search over credit (name, email, URL, ORCID iD, type of entity, type of role and note)",
    )
    creditName: str | None = Field(  # noqa: N815
        None,
        description="Exact search for name of credited entity",
    )
    creditTypeRole: str | None = Field(  # noqa: N815
        None,
        description="Exact search for role of credited entity",
    )
    creditTypeEntity: str | None = Field(  # noqa: N815
        None,
        description="Exact search for type of credited entity",
    )
    creditOrcidID: str | None = Field(  # noqa: N815
        None,
        description="Exact search for ORCID iD of credited entity (**must** be quoted)",
    )
    publication: str | None = Field(
        None,
        description=(
            "Fuzzy search over publication (DOI, PMID, PMCID, publication type and tool version) (quoted as needed)"
        ),
    )
    publicationID: str | None = Field(  # noqa: N815
        None,
        description="Exact search for publication ID (DOI, PMID or PMCID) (**must** be quoted)",
    )
    publicationType: str | None = Field(  # noqa: N815
        None,
        description="Exact search for publication type",
    )
    publicationVersion: str | None = Field(  # noqa: N815
        None,
        description="Exact search for tool version associated with a publication (**must** be quoted)",
    )
    link: str | None = Field(
        None,
        description="Fuzzy search over general link (URL, type and note) (quote as needed)",
    )
    linkType: str | None = Field(  # noqa: N815
        None,
        description="Exact search for type of information found at a link",
    )
    documentation: str | None = Field(
        None,
        description="Fuzzy search over documentation link (URL, type and note) (quote as needed)",
    )
    documentationType: str | None = Field(  # noqa: N815
        None,
        description="Exact search for type of documentation",
    )
    download: str | None = Field(
        None,
        description="Fuzzy search over download link (URL, type, version and note) (quote as needed)",
    )
    downloadType: str | None = Field(  # noqa: N815
        None,
        description="Exact search for type of download",
    )
    downloadVersion: str | None = Field(  # noqa: N815
        None,
        description="Exact search for tool version associated with a download (**must** be quoted)",
    )
    otherID: str | None = Field(  # noqa: N815
        None,
        description="Fuzzy search over alternate tool IDs (ID value, type of ID and version)",
    )
    otherIDValue: str | None = Field(  # noqa: N815
        None,
        description="Exact search for value of alternate tool ID (**must** be quoted)",
    )
    otherIDType: str | None = Field(  # noqa: N815
        None,
        description="Exact search for type of alternate tool ID",
    )
    otherIDVersion: str | None = Field(  # noqa: N815
        None,
        description="Exact search for tool version associated with an alternate ID (**must** be quoted)",
    )
    question_uuid: str | None = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question.",
    )


class BioToolsQueryBuilder(BaseQueryBuilder):
    """A class for building an BioToolsQuery object."""

    def create_runnable(
        self,
        query_parameters: "BioToolsQueryParameters",
        conversation: "Conversation",
    ) -> Callable:
        """Create a runnable object for executing queries.

        Create runnable using the LangChain `create_structured_output_runnable`
        method.

        Args:
        ----
            query_parameters: A Pydantic data model that specifies the fields of
                the API that should be queried.

            conversation: A BioChatter conversation object.

        Returns:
        -------
            A Callable object that can execute the query.

        """
        return create_structured_output_runnable(
            output_schema=query_parameters,
            llm=conversation.chat,
            prompt=self.structured_output_prompt,
        )

    def parameterise_query(
        self,
        question: str,
        conversation: "Conversation",
    ) -> list[BioToolsQueryParameters]:
        """Generate an BioToolsQuery object.

        Generate a BioToolsQuery object based on the given question, prompt,
        and BioChatter conversation. Uses a Pydantic model to define the API
        fields.  Creates a runnable that can be invoked on LLMs that are
        qualified to parameterise functions.

        Args:
        ----
            question (str): The question to be answered.

            conversation: The conversation object used for parameterising the
                BioToolsQuery.

        Returns:
        -------
            BioToolsQueryParameters: the parameterised query object (Pydantic
                model)

        """
        runnable = self.create_runnable(
            query_parameters=BioToolsQueryParameters,
            conversation=conversation,
        )
        oncokb_call_obj = runnable.invoke(
            {
                "input": f"Answer:\n{question} based on:\n {BIOTOOLS_QUERY_PROMPT}",
            },
        )
        oncokb_call_obj.question_uuid = str(uuid.uuid4())
        return [oncokb_call_obj]


class BioToolsFetcher(BaseFetcher):
    """A class for retrieving API results from BioTools.

    Retrieves API results given a parameterized BioToolsQuery.
    """

    def __init__(self, api_token: str = "demo") -> None:  # noqa: S107
        """Initialise the BioToolsFetcher.

        Args:
        ----
            api_token: The API token for the BioTools API.

        """
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json",
        }
        self.base_url = "https://bio.tools/api"

    def fetch_results(
        self,
        request_data: list[BioToolsQueryParameters],
        retries: int | None = 3,  # noqa: ARG002
    ) -> str:
        """Submit the BioTools query and fetch the results directly.

        No multi-step procedure, thus no wrapping of submission and retrieval in
        this case.

        Args:
        ----
            request_data: List of BioToolsQuery objects (Pydantic models)
                containing the BioTools query parameters.

            retries: The number of retries to fetch the results.

        Returns:
        -------
            str: The results of the BioTools query.

        """
        # For now, we only use the first query in the list
        query = request_data[0]

        # Submit the query and get the URL
        params = query.dict(exclude_unset=True)
        endpoint = params.pop("endpoint")
        params.pop("question_uuid")
        full_url = f"{self.base_url}/{endpoint}"
        response = requests.get(full_url, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()

        # Fetch the results from the URL
        results_response = requests.get(response.url, headers=self.headers, timeout=30)
        results_response.raise_for_status()

        return results_response.text


class BioToolsInterpreter(BaseInterpreter):
    """A class for interpreting BioTools results."""

    def summarise_results(
        self,
        question: str,
        conversation_factory: Callable,
        response_text: str,
    ) -> str:
        """Extract the answer from the BLAST results.

        Args:
        ----
            question (str): The question to be answered.
            conversation_factory: A BioChatter conversation object.
            response_text (str): The response.text returned by bio.tools.

        Returns:
        -------
            str: The extracted answer from the BLAST results.

        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a world class bioinformatician who knows "
                    "everything about bio.tools packages and the "
                    "bioinformatics ecosystem. Your task is to interpret "
                    "results from BioTools API calls and summarise "
                    "them for the user.",
                ),
                ("user", "{input}"),
            ],
        )
        summary_prompt = BIOTOOLS_SUMMARY_PROMPT.format(
            question=question,
            context=response_text,
        )
        output_parser = StrOutputParser()
        conversation = conversation_factory()
        chain = prompt | conversation.chat | output_parser
        return chain.invoke({"input": {summary_prompt}})
