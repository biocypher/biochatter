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
try:
    args = parser.parse_args()
except:
    # Debug
    class Args:
        def __init__(self, input_path, output_path):
            self.input_path = input_path
            self.output_path = output_path

    # Use the Args class to create a mock args object
    args = Args(
        "/Users/slobentanzer/Downloads/2023.11.06.565928v1.full.pdf",
        "poisoning-kgs-with-llms_gpt3.mp3",
    )


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

podcaster = Podcaster(document, "gpt-3.5-turbo")
podcaster.generate_podcast(characters_per_paragraph=8000)

if os.path.splitext(args.output_path)[1] == ".mp3":
    podcaster.podcast_to_file(args.output_path, model="tts-1-hd")
else:
    with open(args.output_path, "w") as f:
        f.write(podcaster.podcast_to_text())
