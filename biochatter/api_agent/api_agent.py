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

from typing import Optional
import os

from .blast import BlastFetcher, BlastQueryBuilder, BlastQuery


## Agent class
class APIAgent:
    """
    A class to interact with the BLAST tool for querying and fetching results.

    Attributes:
        conversation_factory (callable): A function used to create a
            BioChatter conversation.

        blast_result_path (str): The path to save BLAST results.

        blast_prompt_path (str): The path to the BLAST prompt file.

        builder (BlastQueryBuilder): An instance to build BLAST queries.

        fetcher (BlastFetcher): An instance to fetch BLAST results.
    """

    def __init__(self, conversation_factory: callable):
        self.conversation_factory = conversation_factory
        self.blast_result_path = ".blast"
        self.blast_prompt_path = "docs/api_agent/BLAST_tool/persistent_files/api_documentation/BLAST.txt"
        self.builder = BlastQueryBuilder()
        self.fetcher = BlastFetcher()
        self.final_answer = None
        self.error = None

        os.makedirs(self.blast_result_path, exist_ok=True)

    def generate_blast_query(self, question: str) -> Optional[BlastQuery]:
        try:
            conversation = self.conversation_factory()
            return self.builder.generate_blast_query(
                question, self.blast_prompt_path, conversation
            )
        except Exception as e:
            print(f"Error generating BLAST query: {e}")
            return None

    def submit_blast_query(self, blast_query: BlastQuery) -> Optional[str]:
        try:
            return self.builder.submit_blast_query(blast_query)
        except Exception as e:
            print(f"Error submitting BLAST query: {e}")
            return None

    def fetch_blast_results(
        self, question_uuid: str, rid: str
    ) -> Optional[str]:
        try:
            return self.fetcher.fetch_and_save_blast_results(
                question_uuid, rid, self.blast_result_path, 100
            )
        except Exception as e:
            print(f"Error fetching BLAST results: {e}")
            return None

    def extract_answer(
        self, question: str, blast_file_name: str
    ) -> Optional[str]:
        try:
            file_path = os.path.join(self.blast_result_path, blast_file_name)
            return self.fetcher.answer_extraction(question, file_path, 100)
        except Exception as e:
            print(f"Error extracting answer: {e}")
            return None

    def execute(self, question: str):
        """
        Wrapper that uses class methods to execute the API agent logic. Consists
        of 1) query generation, 2) query submission, 3) results fetching, and
        4) answer extraction. The final answer is stored in the final_answer
        attribute.

        Args:
            question (str): The question to be answered.
        """
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
        blast_file_name = self.fetch_blast_results(
            blast_query.question_uuid, rid
        )
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
