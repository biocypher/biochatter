import os
import pytest

###
### TO DO 
###
### Continue to refactor the code in biochatter/api_agent.py, and test it here.


import unittest
from biochatter.api_agent import BlastQueryBuilder, BlastQuery, llm  # Adjust the import as necessary

class TestBlastQueryBuilder(unittest.TestCase):
  
  def setUp(self):
    self.builder = BlastQueryBuilder()
      
  def test_blast_structured_output_prompt(self):
    builder = BlastQueryBuilder()
    prompt_template = builder.BLAST_structured_output_prompt
    self.assertIsNotNone(prompt_template)
      
  def test_read_blast_prompt(self):
    builder = BlastQueryBuilder()
    blast_prompt_path = "docs/api_agent/BLAST_tool/persistent_files/api_documentation/BLAST.txt"
    blast_promot = builder.read_blast_promot(blast_prompt_path) 
    self.assertIsNotNone(blast_promot)
    
  def test_create_runnable(self):
    builder = BlastQueryBuilder()
    blast_prompt_path = "docs/api_agent/BLAST_tool/persistent_files/api_documentation/BLAST.txt"
    blast_promot = builder.read_blast_promot(blast_prompt_path) 
    self.assertIsNotNone(blast_promot)
    runnable = builder.create_runnable(llm, BlastQuery)
    self.assertIsNotNone(runnable)
    
  def test_generate_blast_query(self):
    builder = BlastQueryBuilder()
    blast_prompt_path = "docs/api_agent/BLAST_tool/persistent_files/api_documentation/BLAST.txt"
    question = "Which organism does the DNA sequence come from:TTCATCGGTCTGAGCAGAGGATGAAGTTGCAAATGATGCAAGCAAAACAGCTCAAAGATGAAGAGGAAAAGGCTATACACAACAGGAGCAATGTAGATACAGAAGGT"
    blast_query = builder.generate_blast_query(question, blast_prompt_path, llm)
    self.assertIsNotNone(blast_query)
    # query = builder.submit_blast_query("Which organism does the DNA sequence come from:TTCATCGGTCTGAGCAGAGGATGAAGTTGCAAATGATGCAAGCAAAACAGCTCAAAGATGAAGAGGAAAAGGCTATACACAACAGGAGCAATGTAGATACAGAAGGT")
    # self.assertIsNotNone(query)
  def test_execute_blast_query(self):
    builder = BlastQueryBuilder()
    blast_prompt_path = "docs/api_agent/BLAST_tool/persistent_files/api_documentation/BLAST.txt"
    question = "Which organism does the DNA sequence come from:TTCATCGGTCTGAGCAGAGGATGAAGTTGCAAATGATGCAAGCAAAACAGCTCAAAGATGAAGAGGAAAAGGCTATACACAACAGGAGCAATGTAGATACAGAAGGT"
    blast_query = builder.generate_blast_query(question, blast_prompt_path, llm)
    self.assertIsNotNone(blast_query)
    print(blast_query)
    rid = builder.submit_blast_query(blast_query)    # self.assertIsNotNone(query)