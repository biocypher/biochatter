from typing import Optional
from urllib.parse import urlencode
from collections.abc import Callable
import re
import time
import uuid

from pydantic import Field, BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.openai_functions import create_structured_output_runnable
import requests

from biochatter.llm_connect import Conversation
from .abc import BaseFetcher, BaseInterpreter, BaseQueryBuilder

BLAST_QUERY_PROMPT = """
You are a world class algorithm for creating queries in structured formats. Your task is to use NCBI Web APIs to answer genomic questions.

For questions about DNA sequences (other than genome alignments) you can use BLAST by: "[https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD={Put|Get}&PROGRAM=blastn&MEGABLAST=on&DATABASE=nt&FORMAT_TYPE={XML|Text}&QUERY={sequence}&HITLIST_SIZE={max_hit_size}]".
BLAST maps a specific DNA {sequence} to DNA sequences among different specices.
Use it if you need information about a DNA sequence, see example questions 1.
You need to first PUT the BLAST request and then GET the results using the RID returned by PUT.

Example Question 1: Which organism does this DNA sequence come from: AGGGGCAGCAAACACCGGGACACACCCATTCGTGCACTAATCAGAAACTTTTTTTTCTCAAATAATTCAAACAATCAAAATTGGTTTTTTCGAGCAAGGTGGGAAATTTTTCGAT
Use BLASTn for such a question -> [https://blast.ncbi.nlm.nih.gov/Blast.cgi??CMD=Put&PROGRAM=blastn&DATABASE=nt&QUERY=AGGGGCAGCAAACACCGGGACACACCCATTCGTGCACTAATCAGAAACTTTTTTTTCTCAAATAATTCAAACAATCAAAATTGGTTTTTTCGAGCAAGGTGGGAAATTTTTCGAT&FORMAT_TYPE=Text&MEGABLAST=on&HITLIST_SIZE=15&email=noah.bruderer%40uib.no]

For questions about protein sequences you can also use BLAST, but this time by: [https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Put&PROGRAM=blastp&DATABASE=nr&FORMAT_TYPE=XML&QUERY=sequence&HITLIST_SIZE=max_hit_size]
Example Question 2: What do you find about protein sequence: MEEPQSDPSV
Use BLASTp for such a question -> https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Put&PROGRAM=blastp&DATABASE=nr&FORMAT_TYPE=XML&QUERY=MEEPQSDPSV&HITLIST_SIZE=10"
"""

BLAST_SUMMARY_PROMPT = """
        You have to answer this question in a clear and concise manner: {question} Be factual!\n\
        If you are asked what organism a specific sequence belongs to, check the 'Hit_def' fields. If you find a synthetic construct or predicted entry, move to the next one and look for an organism name.\n\
        Try to use the hits with the best identity score to answer the question. If it is not possible, move to the next one.\n\
        Be clear, and if organism names are present in ANY of the results, please include them in the answer. Do not make up information and mention how relevant the found information is based on the identity scores.\n\
        Use the same reasoning for any potential BLAST results. If you find information that is manually curated, please use it and state it. You may also state other results, but always include the context.\n\
        Based on the information given here:\n\
        {context}
        """


class BlastQueryParameters(BaseModel):
    """

    BlastQuery is a Pydantic model for the parameters of a BLAST query request,
    used for configuring and sending a request to the NCBI BLAST query API. The
    fields are dynamically configured by the LLM based on the user's question.

    """

    url: Optional[str] = Field(
        default="https://blast.ncbi.nlm.nih.gov/Blast.cgi?",
        description="ALWAYS USE DEFAULT, DO NOT CHANGE",
    )
    cmd: Optional[str] = Field(
        default="Put",
        description="Command to execute, 'Put' for submitting query, 'Get' for retrieving results.",
    )
    program: Optional[str] = Field(
        default="blastn",
        description="BLAST program to use, e.g., 'blastn' for nucleotide-nucleotide BLAST, 'blastp' for protein-protein BLAST.",
    )
    database: Optional[str] = Field(
        default="nt",
        description="Database to search, e.g., 'nt' for nucleotide database, 'nr' for non redundant protein database, pdb the Protein Data Bank database, which is used specifically for protein structures, 'refseq_rna' and 'refseq_genomic': specialized databases for RNA sequences and genomic sequences",
    )
    query: Optional[str] = Field(
        None,
        description="Nucleotide or protein sequence for the BLAST or blat query, make sure to always keep the entire sequence given.",
    )
    format_type: Optional[str] = Field(
        default="Text",
        description="Format of the BLAST results, e.g., 'Text', 'XML'.",
    )
    rid: Optional[str] = Field(
        None, description="Request ID for retrieving BLAST results."
    )
    other_params: Optional[dict] = Field(
        default={"email": "user@example.com"},
        description="Other optional BLAST parameters, including user email.",
    )
    max_hits: Optional[int] = Field(
        default=15,
        description="Maximum number of hits to return in the BLAST results.",
    )
    sort_by: Optional[str] = Field(
        default="score",
        description="Criterion to sort BLAST results by, e.g., 'score', 'evalue'.",
    )
    megablast: Optional[str] = Field(
        default="on", description="Set to 'on' for human genome alignemnts"
    )
    question_uuid: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question.",
    )
    full_url: Optional[str] = Field(
        default="TBF",
        description="Full URL to be used to submit the BLAST query",
    )


class BlastQueryBuilder(BaseQueryBuilder):
    """A class for building a BlastQuery object."""

    def create_runnable(
        self,
        query_parameters: "BlastQueryParameters",
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
    ) -> BlastQueryParameters:
        """
        Generates a BlastQuery object based on the given question, prompt, and
        BioChatter conversation. Uses a Pydantic model to define the API fields.
        Creates a runnable that can be invoked on LLMs that are qualified to
        parameterise functions.

        Args:
            question (str): The question to be answered.

            conversation: The conversation object used for parameterising the
                BlastQuery.

        Returns:
            BlastQuery: the parameterised query object (Pydantic model)
        """
        runnable = self.create_runnable(
            query_parameters=BlastQueryParameters,
            conversation=conversation,
        )
        blast_call_obj = runnable.invoke(
            {"input": f"Answer:\n{question} based on:\n {BLAST_QUERY_PROMPT}"}
        )
        blast_call_obj.question_uuid = str(uuid.uuid4())
        return blast_call_obj


class BlastFetcher(BaseFetcher):
    """
    A class for retrieving API results from BLAST given a parameterised
    BlastQuery.

    TODO add a limit of characters to be returned from the response.text?
    """

    def _submit_query(self, request_data: BlastQueryParameters) -> str:
        """Function to POST the BLAST query and retrieve RID.
        It submits the structured BlastQuery obj and return the RID.

        Args:
            request_data: BlastQuery object containing the BLAST query
                parameters.
        Returns:
            str: The Request ID (RID) for the submitted BLAST query.
        """
        data = {
            "CMD": request_data.cmd,
            "PROGRAM": request_data.program,
            "DATABASE": request_data.database,
            "QUERY": request_data.query,
            "FORMAT_TYPE": request_data.format_type,
            "MEGABLAST": request_data.megablast,
            "HITLIST_SIZE": request_data.max_hits,
        }
        # Include any other_params if provided
        if request_data.other_params:
            data.update(request_data.other_params)
        # Make the API call
        query_string = urlencode(data)
        # Combine base URL with the query string
        full_url = f"{request_data.url}?{query_string}"
        # Print the full URL
        request_data.full_url = full_url
        print("Full URL built by retriever:\n", request_data.full_url)
        response = requests.post(request_data.url, data=data)
        response.raise_for_status()
        # Extract RID from response
        print(response)
        match = re.search(r"RID = (\w+)", response.text)
        if match:
            return match.group(1)
        else:
            raise ValueError("RID not found in BLAST submission response.")

    def _fetch_results(
        self,
        rid: str,
        question_uuid: str,
        retries: int = 10000,
    ):
        """SECOND function to be called for a BLAST query.
        Will look for the RID to fetch the data
        """
        ###
        ###    TO DO: Implement logging for all BLAST queries
        ###
        # log_question_uuid_json(request_data.question_uuid,question, file_name, log_file_path,request_data.full_url)
        base_url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
        check_status_params = {
            "CMD": "Get",
            "FORMAT_OBJECT": "SearchInfo",
            "RID": rid,
        }
        get_results_params = {
            "CMD": "Get",
            "FORMAT_TYPE": "XML",
            "RID": rid,
        }

        # Check the status of the BLAST job
        for attempt in range(retries):
            status_response = requests.get(base_url, params=check_status_params)
            status_response.raise_for_status()
            status_text = status_response.text
            print("evaluating status")
            if "Status=WAITING" in status_text:
                print(f"{question_uuid} results not ready, waiting...")
                time.sleep(15)
            elif "Status=FAILED" in status_text:
                raise RuntimeError("BLAST query FAILED.")
            elif "Status=UNKNOWN" in status_text:
                raise RuntimeError("BLAST query expired or does not exist.")
            elif "Status=READY" in status_text:
                if "ThereAreHits=yes" in status_text:
                    print(f"{question_uuid} results are ready, retrieving.")
                    results_response = requests.get(
                        base_url, params=get_results_params
                    )
                    results_response.raise_for_status()
                    # Save the results to a file
                    return results_response.text
                else:
                    return "No hits found"
        if attempt == retries - 1:
            raise TimeoutError(
                "Maximum attempts reached. Results may not be ready."
            )

    def fetch_results(
        self, query_model: BlastQueryParameters, retries: int = 20
    ) -> str:
        """
        Submit request and fetch results from BLAST API. Wraps individual
        submission and retrieval of results.

        Args:
            query_model: the Pydantic model of the query

            retries: the number of maximum retries

        Returns:
            str: the result from the BLAST API
        """
        rid = self._submit_query(request_data=query_model)
        return self._fetch_results(
            rid=rid,
            question_uuid=query_model.question_uuid,
            retries=retries,
        )


class BlastInterpreter(BaseInterpreter):
    def summarise_results(
        self,
        question: str,
        conversation_factory: Callable,
        response_text: str,
    ) -> str:
        """
        Function to extract the answer from the BLAST results.

        Args:
            question (str): The question to be answered.
            conversation_factory: A BioChatter conversation object.
            response_text (str): The response.text returned by NCBI.

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
        summary_prompt = BLAST_SUMMARY_PROMPT.format(
            question=question, context=response_text
        )
        output_parser = StrOutputParser()
        conversation = conversation_factory()
        chain = prompt | conversation.chat | output_parser
        answer = chain.invoke({"input": {summary_prompt}})
        return answer
