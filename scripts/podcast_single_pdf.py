#!/usr/bin/env python3

import argparse
import os
from biochatter.podcast import Podcaster
from biochatter.vectorstore import DocumentReader

# Create the parser
parser = argparse.ArgumentParser(
    description="Generate a podcast from a document."
)

# Add the arguments
parser.add_argument(
    "input_path", type=str, help="The path to the input document."
)
parser.add_argument(
    "output_path", type=str, help="The path to the output document."
)


# Parse the arguments
args = parser.parse_args()
args.input_path = os.path.abspath(args.input_path)
args.output_path = os.path.abspath(args.output_path)

# Echo the arguments
print("Input path: " + args.input_path)
print("Output path: " + args.output_path)

# Test wether input file exists
if not os.path.isfile(args.input_path):
    print("Input file does not exist.")
    exit(1)

# Create output file directory if it does not exist
if not os.path.isdir(os.path.dirname(args.output_path)):
    os.makedirs(os.path.dirname(args.output_path))

# Create the podcast
reader = DocumentReader()
document = reader.load_document(args.input_path)

podcaster = Podcaster(document)
podcaster.generate_podcast(characters_per_paragraph=5000)

with open(args.output_path, "w") as f:
    f.write(podcaster.podcast_to_text())
