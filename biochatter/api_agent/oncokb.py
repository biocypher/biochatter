from typing import Optional
from urllib.parse import urlencode
from collections.abc import Callable
import re
import time
import uuid

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.openai_functions import create_structured_output_runnable
import requests

from biochatter.llm_connect import Conversation
from .abc import BaseQueryBuilder, BaseFetcher, BaseInterpreter

ONCOKB_QUERY_PROMPT = """
You are a world class algorithm for creating queries in structured formats. Your task is to use OncoKB Web APIs to answer genomic questions.

For questions about genomic alterations, you can use the OncoKB API by providing the appropriate parameters based on the type of query.

You have to extract the appropriate information out of the 
Examples:
1. To annotate mutations by protein change, use the endpoint /annotate/mutations/byProteinChange with parameters like hugoSymbol, alteration, tumorType, etc.
2. To annotate copy number alterations, use the endpoint /annotate/copyNumberAlterations with parameters like hugoSymbol, copyNameAlterationType, tumorType, etc.

Use these formats to generate queries based on the question provided.
"""


ONCOKB_SUMMARY_PROMPT = """
        You have to answer this question in a clear and concise manner: {question} Be factual!\n\
        If you are asked what organism a specific sequence belongs to, check the 'Hit_def' fields. If you find a synthetic construct or predicted entry, move to the next one and look for an organism name.\n\
        Try to use the hits with the best identity score to answer the question. If it is not possible, move to the next one.\n\
        Be clear, and if organism names are present in ANY of the results, please include them in the answer. Do not make up information and mention how relevant the found information is based on the identity scores.\n\
        Use the same reasoning for any potential BLAST results. If you find information that is manually curated, please use it and state it. You may also state other results, but always include the context.\n\
        Based on the information given here:\n\
        {context}
        """

class OncoKBQueryParameters(BaseModel):
    base_url: str = Field(
        default="https://demo.oncokb.org/api/v1",
        description="Base URL for the OncoKB API. Default is the demo instance."
    )
    endpoint: str = Field(
        ..., description="Specific API endpoint to hit. Example: 'annotate/mutations/byProteinChange'."
    )
    referenceGenome: Optional[str] = Field(
        default="GRCh37", description="Reference genome, either GRCh37 or GRCh38. The default is GRCh37."
    )
    hugoSymbol: Optional[str] = Field(
        None, description="The gene symbol used in Human Genome Organisation. Example: BRAF."
    )
    entrezGeneId: Optional[int] = Field(
        None, description="The entrez gene ID. Higher priority than hugoSymbol. Example: 673."
    )
    tumorType: Optional[str] = Field(
        None, description="OncoTree(http://oncotree.info) tumor type name. The field supports OncoTree Code, OncoTree Name and OncoTree Main type. Example: Melanoma."
    )
    alteration: Optional[str] = Field(
        None, description="Protein Change. Example: V600E."
    )
    consequence: Optional[str] = Field(
        None, description="Consequence. Example: missense_variant."
    )
    proteinStart: Optional[int] = Field(
        None, description="Protein Start. Example: 600."
    )
    proteinEnd: Optional[int] = Field(
        None, description="Protein End. Example: 600."
    )
    copyNameAlterationType: Optional[str] = Field(
        None, description="Copy number alteration type. Available types: AMPLIFICATION, DELETION, GAIN, LOSS."
    )
    structuralVariantType: Optional[str] = Field(
        None, description="Structural variant type. Available values: DELETION, TRANSLOCATION, DUPLICATION, INSERTION, INVERSION, FUSION, UNKNOWN."
    )
    isFunctionalFusion: Optional[bool] = Field(
        default=False, description="Whether it is a functional fusion. Default value: false."
    )
    hugoSymbolA: Optional[str] = Field(
        None, description="The gene symbol A used in Human Genome Organisation. Example: ABL1."
    )
    entrezGeneIdA: Optional[int] = Field(
        None, description="The entrez gene ID A. Higher priority than hugoSymbolA. Example: 25."
    )
    hugoSymbolB: Optional[str] = Field(
        None, description="The gene symbol B used in Human Genome Organisation. Example: BCR."
    )
    entrezGeneIdB: Optional[int] = Field(
        None, description="The entrez gene ID B. Higher priority than hugoSymbolB. Example: 613."
    )
    genomicLocation: Optional[str] = Field(
        None, description="Genomic location. Example: 7,140453136,140453136,A,T."
    )
    hgvsg: Optional[str] = Field(
        None, description="HGVS genomic format. Example: 7:g.140453136A>T."
    )
    question_uuid: Optional[str] = Field(
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
        """
        Creates a runnable object for executing queries using the LangChain
        `create_structured_output_runnable` method.

        Args:
            query_parameters: A Pydantic data model that specifies the fields of
                the API that should be queried.

            conversation: A BioChatter conversation object.

        Returns:
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
    ) -> OncoKBQueryParameters:
        """
        Generates an OncoKBQuery object based on the given question, prompt, and
        BioChatter conversation. Uses a Pydantic model to define the API fields.
        Creates a runnable that can be invoked on LLMs that are qualified to
        parameterise functions.

        Args:
            question (str): The question to be answered.

            conversation: The conversation object used for parameterising the
                OncoKBQuery.

        Returns:
            OncoKBQueryParameters: the parameterised query object (Pydantic model)
        """
        runnable = self.create_runnable(
            query_parameters=OncoKBQueryParameters,
            conversation=conversation,
        )
        oncokb_call_obj = runnable.invoke(
            {"input": f"Answer:\n{question} based on:\n {ONCOKB_QUERY_PROMPT}"}
        )
        oncokb_call_obj.question_uuid = str(uuid.uuid4())
        return oncokb_call_obj
    
    

class OncoKBFetcher(BaseFetcher):
    """
    A class for retrieving API results from OncoKB given a parameterized
    OncoKBQuery.
    """

    def __init__(self, api_token= 'demo'):
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json"
        }
        self.base_url = "https://demo.oncokb.org/api/v1"

    def submit_query(self, request_data: OncoKBQueryParameters) -> str:
        """Function to submit the OncoKB query and retrieve the URL.
        It submits the structured OncoKBQuery obj and returns the full URL.

        Args:
            request_data: OncoKBQuery object containing the OncoKB query
                parameters.
        Returns:
            str: The full URL for the submitted OncoKB query.
        """
        params = request_data.dict(exclude_unset=True)
        endpoint = params.pop('endpoint')
        params.pop('question_uuid')
        full_url = f"{self.base_url}/{endpoint}"
        print(full_url)
        response = requests.get(full_url, headers=self.headers, params=params)
        response.raise_for_status()
        print(response.url)
        return response.url

    def fetch_and_save_results(
        self,
        question_uuid: uuid,
        query_return: str,
        save_path: str,
        max_attempts: int = 10000,
    ):
        """Function to fetch the results of the OncoKB query and save them
        to a .oncokb file.
        """
        file_name = f"OncoKB_results_{question_uuid}.oncokb"
        
        if not save_path.endswith("/"):
            save_path += "/"
        
        response = requests.get(query_return, headers=self.headers)
        response.raise_for_status()
        
        with open(f"{save_path}{file_name}", "w") as file:
            file.write(response.text)
        
        print(f"Results saved in {file_name}")
        return file_name
    

class OncoKBInterpreter(BaseInterpreter):
    def summarise_results(
        self,
        question: str,
        conversation_factory: Callable,
        file_path: str,
        n_lines: int,
    ) -> str:
        """
        Function to extract the answer from the BLAST results.

        Args:
            question (str): The question to be answered.
            conversation_factory: A BioChatter conversation object.
            file_path (str): The path to the BLAST results file.
            n_lines (int): The number of lines to read from the file.

        Returns:
            str: The extracted answer from the BLAST results.

        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a world class molecular biologist who knows everything about NCBI and BLAST results.",
                ),
                ("user", "{input}"),
            ]
        )

        context = self.read_first_n_lines(file_path, n_lines)
        summary_prompt = ONCOKB_SUMMARY_PROMPT.format(question=question, context=context)
        output_parser = StrOutputParser()
        conversation = conversation_factory()
        chain = prompt | conversation.chat | output_parser
        answer = chain.invoke({"input": {summary_prompt}})
        return answer

    def read_first_n_lines(self, file_path: str, n_lines: int):
        """
        Reads the first n lines from a file and returns them as a string.

        Args:
            file_path (str): The path to the file.
            n_lines (int): The number of lines to read.

        Returns:
            str: The first n lines from the file as a string.

        Raises:
            FileNotFoundError: If the file is not found.
            Exception: If any other error occurs during file reading.

        """
        try:
            with open(file_path, "r") as file:
                lines = []
                for i in range(n_lines):
                    line = file.readline()
                    if not line:
                        break
                    lines.append(line.strip())
                # to test:
                # more efficient with \n or without?
                return "\n".join(lines)
        except FileNotFoundError:
            return "The file was not found."
        except Exception as e:
            return f"An error occurred: {e}"
