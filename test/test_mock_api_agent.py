import os
import uuid
import requests
from unittest.mock import MagicMock, patch, Mock
import unittest
import uuid
from biochatter.api_agent import BlastQueryBuilder, BlastQuery, BlastFetcher, llm 

class TestBlastQueryBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = BlastQueryBuilder()
        self.llm = llm
        
    def test_create_runnable(self):
      builder = BlastQueryBuilder()
      blast_prompt_path = "docs/api_agent/BLAST_tool/persistent_files/api_documentation/BLAST.txt"
      blast_promot = builder.read_blast_promot(blast_prompt_path) 
      self.assertIsNotNone(blast_promot)
      runnable = builder.create_runnable(llm, BlastQuery)
      self.assertIsNotNone(runnable)
    
    @patch('biochatter.api_agent.create_structured_output_runnable')
    def test_generate_blast_query(self, mock_create_runnable):
      blast_prompt_path = "docs/api_agent/BLAST_tool/persistent_files/api_documentation/BLAST.txt"
      question = "Which organism does the DNA sequence come from:TTCATCGGTCTGAGCAGAGGATGAAGTTGCAAATGATGCAAGCAAAACAGCTCAAAGATGAAGAGGAAAAGGCTATACACAACAGGAGCAATGTAGATACAGAAGGT"
      mock_runnable = Mock()
      mock_create_runnable.return_value = mock_runnable
      mock_blast_call_obj = Mock()
      mock_runnable.invoke.return_value = mock_blast_call_obj
      blast_query = self.builder.generate_blast_query(question, blast_prompt_path, llm)
      print(blast_query.question_uuid)
 

    @patch('requests.post')
    def test_submit_blast_query(self, mock_post):
      blast_query = BlastQuery(
          cmd="blastn",
          program="blastn",
          database="nt",
          query="AGCTG",
          format_type="XML",
          megablast=True,
          max_hits=10,
          url="https://blast.ncbi.nlm.nih.gov/Blast.cgi"
      )
      mock_response = Mock()
      mock_response.text = "RID = 1234"
      mock_post.return_value = mock_response

      rid = self.builder.submit_blast_query(blast_query)
      
      self.assertEqual(rid, "1234")
      print(rid)
      
class TestBlastFetcher(unittest.TestCase):
  @patch('requests.get')
  @patch('builtins.open', new_callable=unittest.mock.mock_open)
  def test_fetch_and_save_blast_results(self, mock_open, mock_get):
    fetcher = BlastFetcher()
    question_uuid = str(uuid.uuid4())
    BLAST_result_path = "docs/api_agent/BLAST_tool/BLAST_response/results"
    blast_query_return = '62YGMDCX013'

    # Mock the GET request response for status checking COULD add not ready response, but it would delay the test
    mock_status_response_ready = Mock()
    mock_status_response_ready.text = "Status=READY\nThereAreHits=yes"

    mock_results_response = Mock()
    mock_results_response.text = "Mock BLAST results"

    # Setup the side effect for status and results checking
    mock_get.side_effect = [mock_status_response_ready, mock_results_response]

    file_name = fetcher.fetch_and_save_blast_results(question_uuid, blast_query_return, BLAST_result_path, 100)
    #assert file name is correct
    self.assertEqual(file_name, f'BLAST_results_{question_uuid}.txt')

    # Verify the file write operations
    expected_path = f'{BLAST_result_path}/{file_name}'
    #assert the paths
    mock_open.assert_called_once_with(expected_path, 'w')
    #assert the correct content is in the file 
    mock_open().write.assert_called_once_with("Mock BLAST results")

    @patch('builtins.open', new_callable=mock_open, read_data="line1\nline2\nline3\nline4\nline5\n")
    @patch.object(BlastFetcher, 'read_first_n_lines')
    def test_answer_extraction(self, mock_parser, mock_llm, mock_prompt, mock_read_first_n_lines, mock_file):
        fetcher = BlastFetcher()
        question = "What organism does this sequence belong to?"
        file_path = 'fake_path.txt'
        n = 3
        mock_read_first_n_lines.return_value = "line1\nline2\nline3"
        mock_prompt.from_messages.return_value = mock_prompt
        mock_parser_instance = mock_parser.return_value
        mock_parser_instance.invoke.return_value = "Mocked Answer"

        result = fetcher.answer_extraction(question, file_path, n)

        self.assertEqual(result, "Mocked Answer")
        mock_read_first_n_lines.assert_called_once_with(file_path, n)
        mock_parser_instance.invoke.assert_called_once()