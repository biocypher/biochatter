### logic

# 1. User asks question req. BLAST to find the answer
# 2. rag_agent receives the question and sends it to the api_agent // api_agent is manually selected 
# 3. api_agent writes query for BLAST specific to the question
    # 3.1 question + BLAST prompt template + BlastQuery are input into 
    # langchain.chains.openai_functions.create_structured_output_runnable -> returns a structured output which is our API call object
# 4. BLAST API call 
    # 4.1 BLAST API call is made using the structured output from 3.1
    # 4.2 BLAST API returns a response which is saved 
    # 
# 5. Read response from BLAST and uses it to answer question
# 6. answer is returned to rag_agent 

###
###     TO DO: Handle all dependencies and imports with poetry 
###

from pydantic import BaseModel, Field
from typing import Optional
import uuid
from urllib.parse import urlencode
import requests
import os 
import re
import time
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.chains.openai_functions import (
    create_structured_output_runnable,
)

###
###     TO DO: IMPLEMENT LLM CALLING LOGIC THROUGH llm_connect.py
###
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name='gpt-4', temperature=0, openai_api_key=openai_api_key)
    
    
class BlastQuery(BaseModel):
    """BlastQuery is a Pydantic model for the BLAST query request.
    A Pydantic model for configuring and sending a request to the NCBI BLAST query API.
    """
    
    url: Optional[str] = Field(
        default="https://blast.ncbi.nlm.nih.gov/Blast.cgi?",
        description="ALWAYS USE DEFAULT, DO NOT CHANGE"
    )
    cmd: Optional[str] = Field(
        default="Put",
        description="Command to execute, 'Put' for submitting query, 'Get' for retrieving results."
    )
    program: Optional[str] = Field(
        default="blastn",
        description="BLAST program to use, e.g., 'blastn' for nucleotide-nucleotide BLAST, 'blastp' for protein-protein BLAST."
    )
    database: Optional[str] = Field(
        default="nt",
        description="Database to search, e.g., 'nt' for nucleotide database, 'nr' for non redundant protein database, pdb the Protein Data Bank database, which is used specifically for protein structures, 'refseq_rna' and 'refseq_genomic': specialized databases for RNA sequences and genomic sequences"
    )
    query: Optional[str] = Field(
        None,
        description="Nucleotide or protein sequence for the BLAST or blat query, make sure to always keep the entire sequence given."
    )
    format_type: Optional[str] = Field(
        default="Text",
        description="Format of the BLAST results, e.g., 'Text', 'XML'."
    )
    rid: Optional[str] = Field(
        None,
        description="Request ID for retrieving BLAST results."
    )
    other_params: Optional[dict] = Field(
        default={"email": "noah.bruderer@uib.no"},
        description="Other optional BLAST parameters, including user email."
    )
    max_hits: Optional[int] = Field(
        default=15,
        description="Maximum number of hits to return in the BLAST results."
    )
    sort_by: Optional[str] = Field(
        default="score",
        description="Criterion to sort BLAST results by, e.g., 'score', 'evalue'."
    )
    megablast: Optional[str] = Field(
        default="on", 
        description="Set to 'on' for human genome alignemnts"
    )
    question_uuid: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question."
    )
    full_url: Optional[str] = Field(
        default='TBF',
        description="Url used for the blast query"
    )

class BlastQueryBuilder(BaseModel):
    """A pydantic class for building a BlastQuery object."""

    @property
    def BLAST_structured_output_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a world class algorithm for extracting information in structured formats.",
            ),
            (
                "human",
                "Use the given format to extract information from the following input: {input}",
            ),
            ("human", "Tip: Make sure to answer in the correct format"),
        ]
    )
    def read_blast_promot(self, BLAST_prompt_file_path: str) -> str:
        try:
            with open(BLAST_prompt_file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
             return "The file was not found at the specified path."
        except Exception as e:
            return "An error occurred while reading the file:", str(e)
    
    def create_runnable(self, llm, blast_query_class) -> callable:
        return create_structured_output_runnable(blast_query_class, llm, self.BLAST_structured_output_prompt)
    
    def generate_blast_query(self, question: str, BLAST_prompt_path: str, llm) -> BlastQuery:
        """
        Generates a BlastQuery object based on the given question, file path, and llm.

        Args:
            question (str): The question to be answered.
            file_path (str): The path to the file containing the BLAST prompt.
            llm: The llm object used for creating the BlastQuery.

        Returns:
            BlastQuery: The generated BlastQuery object.
        """
        runnable = self.create_runnable(llm, BlastQuery)
        BLAST_prompt = self.read_blast_promot(BLAST_prompt_path)
        blast_call_obj = runnable.invoke({"input": f"Answer:\n{question} based on:\n {BLAST_prompt}"})
        blast_call_obj.question_uuid = str(uuid.uuid4())
        return blast_call_obj

    def submit_blast_query(self, request_data: BlastQuery) -> str:
        """Function to POST the BLAST query and retrieve RID.
        It submits the structured BlastQuery obj and return the RID.
        
        Args:
            request_data: BlastQuery object containing the BLAST query parameters.
        Returns:
            str: The Request ID (RID) for the submitted BLAST query.
        """
        data = {
            'CMD': request_data.cmd,
            'PROGRAM': request_data.program,
            'DATABASE': request_data.database,
            'QUERY': request_data.query,
            'FORMAT_TYPE': request_data.format_type,
            'MEGABLAST':request_data.megablast,
            'HITLIST_SIZE':request_data.max_hits,
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

class BlastFetcher(BaseModel):
    def fetch_and_save_blast_results(self, question_uuid: uuid, blast_query_return: str, save_path: str , 
                                    max_attempts: int = 10000):
        """SECOND function to be called for a BLAST query.
        Will look for the RID to fetch the data
        """
        file_name = f'BLAST_results_{question_uuid}.txt'
        ###
        ###    TO DO: Implement logging for all BLAST queries
        ###
        # log_question_uuid_json(request_data.question_uuid,question, file_name, save_path, log_file_path,request_data.full_url)        
        base_url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
        check_status_params = {
            'CMD': 'Get',
            'FORMAT_OBJECT': 'SearchInfo',
            'RID': blast_query_return
        }
        get_results_params = {
            'CMD': 'Get',
            'FORMAT_TYPE': 'XML',
            'RID': blast_query_return
        }
        #checking path: should end with '/'
        if not save_path.endswith('/'):
            save_path += '/'
        # Check the status of the BLAST job
        for attempt in range(max_attempts):
            status_response = requests.get(base_url, params=check_status_params)
            status_response.raise_for_status()
            status_text = status_response.text
            print('evaluating status')
            if 'Status=WAITING' in status_text:
                print(f"{question_uuid} results not ready, waiting...")
                time.sleep(15)
            elif 'Status=FAILED' in status_text:
                with open(f'{save_path}{file_name}', 'w') as file:
                    file.write("BLAST query FAILED.")
            elif 'Status=UNKNOWN' in status_text:
                with open(f'{save_path}{file_name}', 'w') as file:
                    file.write("BLAST query expired or does not exist.")
                raise 
            elif 'Status=READY' in status_text:
                if 'ThereAreHits=yes' in status_text:
                    print(f"{question_uuid} results are ready, retrieving and saving...")
                    results_response = requests.get(base_url, params=get_results_params)
                    results_response.raise_for_status()
                    # Save the results to a file
                    print(f'{save_path}{file_name}')
                    with open(f'{save_path}{file_name}', 'w') as file:
                        file.write(results_response.text)
                    print(f'Results saved in BLAST_results_{question_uuid}.txt')
                    break
                else:
                    with open(f'{save_path}{file_name}', 'w') as file:
                        file.write("No hits found")
                    break
            else:
                print('Unknown status')
                with open(f'{save_path}{file_name}', 'w') as file:
                    file.write("Unknown status")
                break 
        if attempt == max_attempts - 1:
            raise TimeoutError("Maximum attempts reached. Results may not be ready.")
        return file_name


    def read_first_n_lines(self, file_path: str, n: int):
        """
        Reads the first n lines from a file and returns them as a string.

        Args:
            file_path (str): The path to the file.
            n (int): The number of lines to read.

        Returns:
            str: The first n lines from the file as a string.

        Raises:
            FileNotFoundError: If the file is not found.
            Exception: If any other error occurs during file reading.

        """
        try:
            with open(file_path, 'r') as file:
                lines = []
                for i in range(n):
                    line = file.readline()
                    if not line:
                        break
                    lines.append(line.strip())
                #to test:
                #more efficient with \n or without?
                return '\n'.join(lines)
        except FileNotFoundError:
            return "The file was not found."
        except Exception as e:
            return f"An error occurred: {e}"


    def answer_extraction(self, question:str, file_path: str, n: int) -> str:
        """
        Function to extract the answer from the BLAST results.

        Args:
            question (str): The question to be answered.
            file_path (str): The path to the BLAST results file.
            n (int): The number of lines to read from the file.

        Returns:
            str: The extracted answer from the BLAST results.

        """
        prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a world class molecular biologist who knows everything about NCBI and BLAST results."),
        ("user", "{input}")
        ])

        context = self.read_first_n_lines(file_path, n)
        BLAST_file_answer_extractor_prompt = f"""
                You have to answer the question: {question} in a clear and concise manner. Be factual!\n\
                If you are asked what organism a specific sequence belongs to, check the 'Hit_def' fields. If you find a synthetic construct or predicted entry, move to the next one and look for an organism name.\n\
                Try to use the hits with the best identity score to answer the question. If it is not possible, move to the next one.\n\
                Be clear, and if organism names are present in ANY of the results, please include them in the answer. Do not make up information and mention how relevant the found information is based on the identity scores.\n\
                Use the same reasoning for any potential BLAST results. If you find information that is manually curated, please use it and state it. You may also state other results, but always include the context.\n\
                Based on the information given here:\n\
                {context}
                """
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        BLAST_answer = chain.invoke({"input": {BLAST_file_answer_extractor_prompt}})
        return BLAST_answer

## Agent class
class APIAgent:
    """
    A class to interact with the BLAST tool for querying and fetching results.

    Attributes:
        llm (object): The language model to be used.
        blast_result_path (str): The path to save BLAST results.
        blast_prompt_path (str): The path to the BLAST prompt file.
        builder (BlastQueryBuilder): An instance to build BLAST queries.
        fetcher (BlastFetcher): An instance to fetch BLAST results.
    """    
    def __init__(self, llm):
        self.llm = llm
        self.blast_result_path = ".blast"
        self.blast_prompt_path = "docs/api_agent/BLAST_tool/persistent_files/api_documentation/BLAST.txt"
        self.builder = BlastQueryBuilder()
        self.fetcher = BlastFetcher()
        self.final_answer = None
        self.error = None
    
        os.makedirs(self.blast_result_path, exist_ok=True)

    def generate_blast_query(self, question: str) -> Optional[BlastQuery]:
        try:
            return self.builder.generate_blast_query(question, self.blast_prompt_path, self.llm)
        except Exception as e:
            print(f"Error generating BLAST query: {e}")
            return None

    def submit_blast_query(self, blast_query: BlastQuery) -> Optional[str]:
        try:
            return self.builder.submit_blast_query(blast_query)
        except Exception as e:
            print(f"Error submitting BLAST query: {e}")
            return None

    def fetch_blast_results(self, question_uuid: str, rid: str) -> Optional[str]:
        try:
            return self.fetcher.fetch_and_save_blast_results(question_uuid, rid, self.blast_result_path, 100)
        except Exception as e:
            print(f"Error fetching BLAST results: {e}")
            return None

    def extract_answer(self, question: str, blast_file_name: str) -> Optional[str]:
        try:
            file_path = os.path.join(self.blast_result_path, blast_file_name)
            return self.fetcher.answer_extraction(question, file_path, 100)
        except Exception as e:
            print(f"Error extracting answer: {e}")
            return None

    def execute(self, question: str):
        # Generate BLAST query
        blast_query = self.generate_blast_query(question)
        if not blast_query:
            print("Failed to generate BLAST query.")
            self.error = "Failed to generate BLAST query."
            return
        
        print(f"Generated BLAST query: {blast_query}")

        # Submit BLAST query and get RID
        rid = self.submit_blast_query(blast_query)
        if not rid:
            print("Failed to submit BLAST query.")
            self.error = "Failed to submit BLAST query."
            return
        
        print(f"Received RID: {rid}")

        # Fetch BLAST results
        blast_file_name = self.fetch_blast_results(blast_query.question_uuid, rid)
        if not blast_file_name:
            print("Failed to fetch BLAST results.")
            self.error = "Failed to fetch BLAST results."
            return

        # Extract answer from BLAST results
        final_answer = self.extract_answer(question, blast_file_name)
        if not final_answer:
            print("Failed to extract answer from BLAST results.")
            self.error = f"Failed to extract answer from BLAST results."
            return

        if final_answer:
            print(f"Final Answer: {final_answer}")
            self.final_answer = final_answer
            return 