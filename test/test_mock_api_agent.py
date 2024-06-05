import os
import uuid
import requests
from unittest.mock import MagicMock, patch, Mock
import unittest
import uuid
from biochatter.api_agent import BlastQueryBuilder, BlastQuery, llm 

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
      print(blast_query )
      
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