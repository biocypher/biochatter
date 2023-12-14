import pandas as pd
import requests
import json
from word2number import w2n
from xinference.client import Client
from langchain.llms import Xinference
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import math

BENCHMARK_DF_PATH = 'benchmark/benchmark_datasets.csv'
FILE_PATH = "benchmark/results/numeric_qa.csv"

def preprocess_gender_represenation_df(bechmark_df):
    """
    Preprocess a DataFrame column by removing special characters and converting values to integers.

    Args:
        column (pd.Series): The DataFrame column to preprocess.

    Returns:
        pd.Series: The preprocessed DataFrame column.
    """
    # Remove '0;' prefix and any non-alphanumeric characters, then split on semicolons
    # Fetch the data
    dataset_url = bechmark_df[bechmark_df['Name'] == 'biohackathon-2022-project-24']['Access'].to_list()[0]
    response = requests.get(dataset_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Read the content line by line and convert each line from JSON
        json_data = [json.loads(line) for line in response.iter_lines(decode_unicode=True) if line]

    else:
        print(f"Failed to retrieve data: Status code {response.status_code}")
        exit()

    # Now 'df' contains the data from the JSON Lines file

    # Initialize a dictionary to hold counts for each ID
    data = []

    # Open the file and process each line
    for json_obj in json_data:
        # Parse the JSON line
        # Extract the ID
        article_id = json_obj['id']

        # Initialize the counts for this ID if not already present
        counts_per_id = {}
        counts_per_id = {'articleid': article_id, 'text': json_obj['text'], 'n_fem': '0', 'n_male': '0',
                         'n_sample': '0'}

        # Check if 'spans' is in the json object
        if 'spans' in json_obj:
            for span in json_obj['spans']:
                # Update counts based on the label
                if span['label'] == 'n_fem':
                    counts_per_id['n_fem'] += ';' + json_obj['text'][span['start']:span['end']]
                elif span['label'] == 'n_male':
                    counts_per_id['n_male'] += ';' + json_obj['text'][span['start']:span['end']]
                elif span['label'] == 'sample':
                    counts_per_id['n_sample'] += ';' + json_obj['text'][span['start']:span['end']]
        data.append(counts_per_id)
    # Print the results
    df = pd.DataFrame.from_records(data)

    def convert_to_int(value):
        """
        Convert a value to an integer. If the value is in word form, convert it using word2number.
        If conversion fails, return None.

        Args:
            value (str or int): The value to convert.

        Returns:
            int or None: The converted integer or None if conversion fails.
        """
        try:
            return int(value)
        except ValueError:
            try:
                return w2n.word_to_num(value)
            except ValueError:
                return None

    def preprocess_column(column):
        """
        Preprocess a DataFrame column by removing special characters and converting values to integers.

        Args:
            column (pd.Series): The DataFrame column to preprocess.

        Returns:
            pd.Series: The preprocessed DataFrame column.
        """
        # Remove '0;' prefix and any non-alphanumeric characters, then split on semicolons
        processed = (column.str.replace('0;', '')
                     .str.replace('[^a-zA-Z0-9]', '', regex=True)
                     .str.lower()
                     .str.split(';')
                     .explode()
                     .apply(convert_to_int))

        return processed

    # Example DataFrame (Assuming df is already defined)
    # Preprocess the specified columns
    columns_to_process = ['n_fem', 'n_male', 'n_sample']
    for column in columns_to_process:
        df[column] = preprocess_column(df[column])

    # Add a 'total' column and fill 'n_sample' where it's 0 with the sum of 'n_fem' and 'n_male'
    df['total'] = df['n_fem'] + df['n_male']
    df.loc[df['n_sample'] == 0, 'n_sample'] = df['total']

    # Filter rows based on annotation correctness
    correctly_annotated = df[df['total'] == df['n_sample']]
    incorrectly_annotated = df[df['total'] != df['n_sample']]  # Optional: Use if needed

    # Deduplicate based on 'articleid' with a priority for rows with fewer zeroes
    df['zero_count'] = (df.drop('articleid', axis=1) == 0).sum(axis=1)
    df = (df.sort_values(by=['articleid', 'zero_count'])
          .drop_duplicates(subset='articleid', keep='first')
          .drop('zero_count', axis=1)).dropna()

    columns_to_process = ['n_fem', 'n_male', 'n_sample', 'total']
    for column in columns_to_process:
        df[column] = df[column].astype(int)
    # Final DataFrame

    return  df[df['total'] == df['n_sample']]


def prompt_engine():
    response_schema = [ResponseSchema(name="n_fem",
                                      description="number of females referred in the sentence. If you don't know write NA."),
                       ResponseSchema(name="n_male",
                                      description="number of females referred in the sentence. If you don't know write NA."),
                       ResponseSchema(name="n_sample",
                                      description="total number of individuals (male and females) referred in the sentence. If you don't know write NA."),
                       ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schema)
    # The parser that will look for the LLM output in my schema and return it back to me
    format_instructions = output_parser.get_format_instructions()

    return PromptTemplate(
        template="""[INST]Given the following sentence from the user, \n
                    get information about the following number. \n
                    {format_instructions}\n{user_prompt}[/INST]""",
        input_variables=["user_prompt"],
        partial_variables={"format_instructions": format_instructions}
    ), output_parser


def call_llm(prompt, llm, text_sentence):
    _input = prompt.format_prompt(user_prompt=text_sentence)
    output = llm(_input.to_string())
    # print(text_sentence['uniqueid'], output)
    return output


def parse_output(output_parser, output):
    json_output = output_parser.parse(output)
    return json_output


def compute_score(results_dictionary, text_infos, articleid):
    # articleid='989_PMC1480516'
    json_output = results_dictionary[articleid]
    text_sentence = text_infos[articleid]
    predictions = [json_output['n_fem'], json_output['n_male'], json_output['n_sample']]
    predictions = [np.nan if x == 'NA' else x for x in predictions]
    if any(math.isnan(x) for x in predictions if isinstance(x, float)):
        predictions = [-99 if np.isnan(x) else int(x) for x in predictions]
    else:
        predictions = [int(x) for x in predictions]

    ground_truth = [text_sentence['n_fem'], text_sentence['n_male'], text_sentence['n_sample']]
    ground_truth = [0 if np.isnan(x) else int(x) for x in ground_truth]

    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='macro', zero_division=0
    )
    return accuracy, precision, recall, f1


def compute_overall_score(results_dictionary, text_infos):
    results = []
    for articleid in results_dictionary.keys():
        accuracy, precision, recall, f1 = compute_score(results_dictionary, text_infos, articleid)
        results.append([articleid, accuracy, precision, recall, f1])

    results_df = pd.DataFrame(results, columns=['articleid', 'accuracy', 'precision', 'recall', 'f1'])
    articleid_to_text = {value['uniqueid']: value['text'] for key, value in text_infos.items()}
    articleid_to_pmid = {value['uniqueid']: value['articleid'] for key, value in text_infos.items()}

    results_df['text'] = results_df['articleid'].map(articleid_to_text)
    results_df['PMID'] = results_df['articleid'].map(articleid_to_pmid)

    identified_articles = set(results_dictionary.keys())
    all_articles= set(text_infos.keys())
    percentage_retrieved = len(identified_articles)/len(all_articles)
    # Calculating mean of each metric
    mean_accuracy = results_df['accuracy'].mean()
    mean_precision = results_df['precision'].mean()
    mean_recall = results_df['recall'].mean()
    mean_f1 = results_df['f1'].mean()

    return mean_accuracy, mean_precision, mean_recall, mean_f1, percentage_retrieved, results_df


def run_test(bechmark_df, model_uid, results_dictionary, main_url):
    prompt, output_parser = prompt_engine()
    input_df = preprocess_gender_represenation_df(bechmark_df)
    input_df['uniqueid'] = input_df.index.astype(str) + "_" + input_df['articleid'].astype(str)
    text_infos = input_df.to_dict('records')

    llm = Xinference(server_url=main_url,
                     model_uid=model_uid,
                     temperature=0.3
                     )

    for text_sentence in tqdm(text_infos):
        if text_sentence['uniqueid'] not in results_dictionary.keys():
            try:
                output = call_llm(prompt, llm, text_sentence['text'])
                json_output = parse_output(output_parser, output)
                results_dictionary[text_sentence['uniqueid']] = json_output
            except:
                # print("error")
                continue

    text_infos = {item['uniqueid']: item for item in text_infos}
    mean_accuracy, mean_precision, mean_recall, mean_f1, percentage_retrieved, results_df  = compute_overall_score(results_dictionary, text_infos)

    return mean_accuracy, mean_precision, mean_recall, mean_f1, percentage_retrieved, results_df


def main():
    bechmark_df = pd.read_csv(BENCHMARK_DF_PATH)
    main_url = "http://llm.biocypher.org"
    models = Client(main_url).list_models()
    model_uids = {model['model_name']: uid for uid, model in models.items() if model['model_type'] != 'embedding'}
    MODEL_NAMES = list(model_uids.keys())

    results_dictionary = {}
    # Here we specify the model to be used
    MODEL_NAME = MODEL_NAMES[0]
    # model_uid = model_uids[MODEL_NAME]
    mean_accuracy, mean_precision, mean_recall, mean_f1, percentage_retrieved, results_df = run_test(bechmark_df, model_uids[MODEL_NAMES[0]], results_dictionary, main_url)
    # percentage_retrieved = proportion of answers with parsable output / total
    # mean scoring values are calculated from the intersection of the benchmark dataset and the answers with parsable output
    with open(FILE_PATH, "a") as f:
        f.write(f"{MODEL_NAME},{mean_accuracy}, {mean_precision}, {mean_recall}, {mean_f1}, {percentage_retrieved}\n")



if __name__ == '__main__':
    main()

