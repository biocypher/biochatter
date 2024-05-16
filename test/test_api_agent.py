import os
import pytest
from biochatter.api_agent import BLAST_api_query_generator, submit_blast_query, fetch_and_save_blast_results, api_agent, answer_extraction

###
### TO DO 
###
###choose what way to test: the commented out tests are for the individual functions, 
###however they are interdependent so the test_api_agent_functions() tests them in one go.


# def test_BLAST_api_query_generator():
#     query_request = BLAST_api_query_generator(question, BLAST_prompt_file_path)
#     assert query_request is not None
#     assert query_request.query == question.split(":")[1]
#     print(query_request)

# def test_submit_blast_query():
#     query_request = BLAST_api_query_generator(question, BLAST_prompt_file_path)
#     rid = submit_blast_query(query_request)
#     assert rid is not None
#     assert isinstance(rid, str)
#     print(rid)

# def test_fetch_and_save_blast_results():
#     query_request = BLAST_api_query_generator(question, BLAST_prompt_file_path)
#     rid = submit_blast_query(query_request)
#     BLAST_file_name = fetch_and_save_blast_results(query_request, rid, BLAST_result_path, 1000)
#     assert BLAST_file_name is not None
#     assert isinstance(BLAST_file_name, str)
#     assert os.path.isfile(os.path.join(BLAST_result_path, BLAST_file_name))
#     print(BLAST_file_name)

def test_api_agent_functions():
# Define the common variables
    question = "Which organism does the DNA sequence come from:TTCATCGGTCTGAGCAGAGGATGAAGTTGCAAATGATGCAAGCAAAACAGCTCAAAGATGAAGAGGAAAAGGCTATACACAACAGGAGCAATGTAGATACAGAAGGT"
    blast_prompt_path = "docs/api_agent/BLAST_tool/persistent_files/api_documentation/BLAST.txt"
    BLAST_result_dir = "docs/api_agent/BLAST_tool/BLAST_response/"
    base_dir = os.getcwd()
    BLAST_prompt_file_path = os.path.join(base_dir, blast_prompt_path)
    BLAST_result_path = os.path.join(base_dir, BLAST_result_dir)
    
    query_request = BLAST_api_query_generator(question, BLAST_prompt_file_path)
    assert query_request is not None
    assert query_request.query == question.split(":")[1]
    print(query_request)
    
    rid = submit_blast_query(query_request)
    assert rid is not None
    assert isinstance(rid, str)
    print(rid)
    
    BLAST_file_name = fetch_and_save_blast_results(query_request, rid, BLAST_result_path, 1000)
    assert BLAST_file_name is not None
    assert isinstance(BLAST_file_name, str)
    BLAST_result_file_path = os.path.join(BLAST_result_path, BLAST_file_name)
    assert os.path.isfile(BLAST_result_file_path)
    print(BLAST_result_file_path)
    final_answer = answer_extraction(question, BLAST_result_file_path, 150)
    assert isinstance(final_answer, str)
    print(final_answer)
    # return final_answer

# x = test_api_agent_functions()