"""OncoKB API agent."""

import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING

import requests
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from biochatter.api_agent.base.agent_abc import (
    BaseFetcher,
    BaseInterpreter,
    BaseQueryBuilder,
)

if TYPE_CHECKING:
    from biochatter.llm_connect import Conversation


ONCOKB_QUERY_PROMPT = """
You are a world class algorithm for creating queries in structured formats. Your task is to use OncoKB Web APIs to
answer genomic questions.

For questions about genomic alterations, you can use the OncoKB API by providing the appropriate parameters based on the
type of query.

You have to extract the appropriate information out of the
Examples:
1. To annotate mutations by protein change, use the endpoint /annotate/mutations/byProteinChange with parameters like
    hugoSymbol, alteration, tumorType, etc.
2. To annotate copy number alterations, use the endpoint /annotate/copyNumberAlterations with parameters like
    hugoSymbol, copyNameAlterationType, tumorType, etc.

Use these formats to generate queries based on the question provided. Below is more information about the OncoKB API:
OncoKB API Documentation (Summary)

Base URL

https://demo.oncokb.org/api/v1

Endpoints and Parameters

1. Annotate Copy Number Alterations

	•	GET/POST /annotate/copyNumberAlterations
	•	Parameters:
	•	hugoSymbol: The gene symbol. Example: BRAF.
	•	entrezGeneId: The entrez gene ID. Example: 673.
	•	copyNameAlterationType: Copy number alteration type. Example: AMPLIFICATION.
	•	referenceGenome: Reference genome. Default: GRCh37.
	•	tumorType: Tumor type. Example: Melanoma.

2. Annotate Mutations by Genomic Change

	•	GET/POST /annotate/mutations/byGenomicChange
	•	Parameters:
	•	genomicLocation: Genomic location. Example: 7,140453136,140453136,A,T.
	•	referenceGenome: Reference genome. Default: GRCh37.
	•	tumorType: Tumor type. Example: Melanoma.

3. Annotate Mutations by HGVSg

	•	GET/POST /annotate/mutations/byHGVSg
	•	Parameters:
	•	hgvsg: HGVS genomic format. Example: 7:g.140453136A>T.
	•	referenceGenome: Reference genome. Default: GRCh37.
	•	tumorType: Tumor type. Example: Melanoma.

4. Annotate Mutations by Protein Change

	•	GET/POST /annotate/mutations/byProteinChange
	•	Parameters:
	•	hugoSymbol: The gene symbol. Example: BRAF.
	•	entrezGeneId: The entrez gene ID. Example: 673.
	•	alteration: Protein Change. Example: V600E.
	•	consequence: Consequence. Example: missense_variant.
	•	proteinStart: Protein Start. Example: 600.
	•	proteinEnd: Protein End. Example: 600.
	•	referenceGenome: Reference genome. Default: GRCh37.
	•	tumorType: Tumor type. Example: Melanoma.

5. Annotate Structural Variants

	•	GET/POST /annotate/structuralVariants
	•	Parameters:
	•	hugoSymbolA: Gene symbol A. Example: ABL1.
	•	entrezGeneIdA: Entrez gene ID A. Example: 25.
	•	hugoSymbolB: Gene symbol B. Example: BCR.
•	entrezGeneIdB: Entrez gene ID B. Example: 613.
	•	structuralVariantType: Structural variant type. Example: FUSION.
	•	isFunctionalFusion: Whether it is a functional fusion. Default: false.
	•	referenceGenome: Reference genome. Default: GRCh37.
	•	tumorType: Tumor type. Example: Melanoma.

6. Get Curated Genes

	•	GET /utils/allCuratedGenes
	•	Parameters:
	•	version: Data version.
	•	includeEvidence: Include gene summary and background.

7. Get Cancer Gene List

	•	GET /utils/cancerGeneList
	•	Parameters:
	•	version: Data version.
 """


ONCOKB_SUMMARY_PROMPT = """
You have to answer this question in a clear and concise manner: {question} Be factual!\n\
You are a world leading oncologist and molecular biologist who knows everything about OncoKB results.\n\
Do not make up information, only use the provided information and mention how relevant the found information is based on your knowledge about OncoKB\n\
Here is the information relevant to the question found on OncoKB:\n\
{context}
"""


class OncoKBQueryParameters(BaseModel):
    base_url: str = Field(
        default="https://demo.oncokb.org/api/v1",
        description="Base URL for the OncoKB API. Default is the demo instance.",
    )
    endpoint: str = Field(
        ...,
        description="Specific API endpoint to hit. Example: 'annotate/mutations/byProteinChange'.",
    )
    referenceGenome: str | None = Field(
        default="GRCh37",
        description="Reference genome, either GRCh37 or GRCh38. The default is GRCh37.",
    )
    hugoSymbol: str | None = Field(
        None,
        description="The gene symbol used in Human Genome Organisation. Example: BRAF.",
    )
    entrezGeneId: int | None = Field(
        None,
        description="The entrez gene ID. Higher priority than hugoSymbol. Example: 673.",
    )
    tumorType: str | None = Field(
        None,
        description="OncoTree(http://oncotree.info) tumor type name. The field supports OncoTree Code, OncoTree Name and OncoTree Main type. Example: Melanoma.",
    )
    alteration: str | None = Field(
        None,
        description="Protein Change. Example: V600E.",
    )
    consequence: str | None = Field(
        None,
        description="Consequence. Example: missense_variant.",
    )
    proteinStart: int | None = Field(
        None,
        description="Protein Start. Example: 600.",
    )
    proteinEnd: int | None = Field(
        None,
        description="Protein End. Example: 600.",
    )
    copyNameAlterationType: str | None = Field(
        None,
        description="Copy number alteration type. Available types: AMPLIFICATION, DELETION, GAIN, LOSS.",
    )
    structuralVariantType: str | None = Field(
        None,
        description="Structural variant type. Available values: DELETION, TRANSLOCATION, DUPLICATION, INSERTION, INVERSION, FUSION, UNKNOWN.",
    )
    isFunctionalFusion: bool | None = Field(
        default=False,
        description="Whether it is a functional fusion. Default value: false.",
    )
    hugoSymbolA: str | None = Field(
        None,
        description="The gene symbol A used in Human Genome Organisation. Example: ABL1.",
    )
    entrezGeneIdA: int | None = Field(
        None,
        description="The entrez gene ID A. Higher priority than hugoSymbolA. Example: 25.",
    )
    hugoSymbolB: str | None = Field(
        None,
        description="The gene symbol B used in Human Genome Organisation. Example: BCR.",
    )
    entrezGeneIdB: int | None = Field(
        None,
        description="The entrez gene ID B. Higher priority than hugoSymbolB. Example: 613.",
    )
    genomicLocation: str | None = Field(
        None,
        description="Genomic location. Example: 7,140453136,140453136,A,T.",
    )
    hgvsg: str | None = Field(
        None,
        description="HGVS genomic format. Example: 7:g.140453136A>T.",
    )
    question_uuid: str | None = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question.",
    )


class OncoKBQueryBuilder(BaseQueryBuilder):
    """A class for building an OncoKBQuery object."""

    def create_runnable(
        self,
        query_parameters: "OncoKBQueryParameters",
        conversation: "Conversation",
    ) -> Callable:
        """Creates a runnable object for executing queries using the LangChain
        `create_structured_output_runnable` method.

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
    ) -> list[OncoKBQueryParameters]:
        """Generate an OncoKBQuery object.

        Generate based on the given question, prompt, and BioChatter
        conversation. Uses a Pydantic model to define the API fields. Creates a
        runnable that can be invoked on LLMs that are qualified to parameterise
        functions.

        Args:
        ----
            question (str): The question to be answered.

            conversation: The conversation object used for parameterising the
                OncoKBQuery.

        Returns:
        -------
            OncoKBQueryParameters: the parameterised query object (Pydantic model)

        """
        runnable = self.create_runnable(
            query_parameters=OncoKBQueryParameters,
            conversation=conversation,
        )
        oncokb_call_obj = runnable.invoke(
            {"input": f"Answer:\n{question} based on:\n {ONCOKB_QUERY_PROMPT}"},
        )
        oncokb_call_obj.question_uuid = str(uuid.uuid4())
        return [oncokb_call_obj]


class OncoKBFetcher(BaseFetcher):
    """A class for retrieving API results.

    Retrieve from OncoKB given a parameterized OncoKBQuery.
    """

    def __init__(self, api_token="demo"):
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json",
        }
        self.base_url = "https://demo.oncokb.org/api/v1"

    def fetch_results(
        self,
        request_data: list[OncoKBQueryParameters],
        retries: int | None = 3,
    ) -> str:
        """Submit the OncoKB query and fetch the results directly.

        No multi-step procedure, thus no wrapping of submission and retrieval in
        this case.

        Args:
        ----
            request_data: List of OncoKBQuery objects (Pydantic models)
                containing the OncoKB query parameters.

            retries: The number of retries to fetch the results.

        Returns:
        -------
            str: The results of the OncoKB query.

        """
        # For now, we only use the first query in the list
        query = request_data[0]

        # Submit the query and get the URL
        params = query.dict(exclude_unset=True)
        endpoint = params.pop("endpoint")
        params.pop("question_uuid")
        full_url = f"{self.base_url}/{endpoint}"
        response = requests.get(full_url, headers=self.headers, params=params)
        response.raise_for_status()

        # Fetch the results from the URL
        results_response = requests.get(response.url, headers=self.headers)
        results_response.raise_for_status()

        return results_response.text


class OncoKBInterpreter(BaseInterpreter):
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
            response_text (str): The response.text returned by OncoKB.

        Returns:
        -------
            str: The extracted answer from the BLAST results.

        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a world class molecular biologist who knows "
                    "everything about OncoKB and cancer genomics. Your task is "
                    "to interpret results from OncoKB API calls and summarise "
                    "them for the user.",
                ),
                ("user", "{input}"),
            ],
        )
        summary_prompt = ONCOKB_SUMMARY_PROMPT.format(
            question=question,
            context=response_text,
        )
        output_parser = StrOutputParser()
        conversation = conversation_factory()
        chain = prompt | conversation.chat | output_parser
        answer = chain.invoke({"input": {summary_prompt}})
        return answer
