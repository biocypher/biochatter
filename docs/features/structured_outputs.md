# Structured Outputs

## Overview

BioChatter enables you to receive responses from the LLM in a predefined structure, rather than just plain text. This is particularly useful when you need the model's output to conform to a specific schema, making it easier to parse and use in downstream tasks. This is achieved by providing a Pydantic model to the conversation.

Many modern LLMs (especially those from providers like OpenAI, Google, Anthropic) can natively generate outputs that conform to a provided schema. For models that do not natively support structured output, BioChatter attempts to guide the model by appending instructions to the prompt, asking it to generate a JSON object matching the schema.

## Defining a Structure (Pydantic Model)

To define the desired output structure, you use Pydantic's `BaseModel`. This allows you to specify fields, types, and even validation rules for the data you expect from the LLM.

Here's an example of a Pydantic model for gene information:

```python
from pydantic import BaseModel

class GeneInfo(BaseModel):
    gene_symbol: str
    full_name: str
    summary: str
    chromosome_location: str | None = None # Optional field
```

This `GeneInfo` model tells the LLM to provide information about a gene, including its `gene_symbol` (string), `full_name` (string), a `summary` of its function (string), and an optional `chromosome_location` (string).

## Requesting Structured Output

You can request structured output by passing your Pydantic model to the `structured_model` parameter in the `query` method of the `LangChainConversation` class.

Here's how you can set up a conversation and request a structured response for gene information:

```python
from biochatter.llm_connect import LangChainConversation
from pydantic import BaseModel

# Define your Pydantic model
class GeneInfo(BaseModel):
    gene_symbol: str
    full_name: str
    summary: str
    chromosome_location: str | None = None

# Initialize the conversation
convo = LangChainConversation(
    model_provider="google_genai",  # Or any other supported provider
    model_name="gemini-2.0-flash", # Ensure model supports structured output or use fallback
    prompts={},
)

# Set API key if required
convo.set_api_key()

# Make the query, passing the Pydantic model
question = "Provide information about the human gene TP53, including its full name and a summary of its function."
convo.query(question, structured_model=GeneInfo)

# Access the structured output
# The last AI message will contain the JSON string of the structured output
structured_response_json = convo.messages[-1].content
print(structured_response_json)

# You can then parse this JSON string back into your Pydantic model
import json
gene_data = json.loads(structured_response_json)
my_gene_info = GeneInfo(**gene_data)
print(f"Gene Symbol: {my_gene_info.gene_symbol}")
print(f"Full Name: {my_gene_info.full_name}")
print(f"Summary: {my_gene_info.summary}")
if my_gene_info.chromosome_location:
    print(f"Location: {my_gene_info.chromosome_location}")
```

If the LLM natively supports structured outputs (e.g., newer OpenAI, Google models), the `AIMessage` content will typically be a JSON string representation of your Pydantic model. You can then parse this string to get an instance of your model.

## The `wrap_structured_output` Parameter

For models that do not natively support structured output, BioChatter tries to instruct the model to generate the output in the correct JSON format. The `wrap_structured_output` parameter in the `query` method can be helpful in these cases:

```python
convo.query(
    question, # Using the gene question from above
    structured_model=GeneInfo,
    wrap_structured_output=True # Defaults to False
)
```

When `wrap_structured_output=True`, BioChatter explicitly asks the non-native model to wrap its JSON output in \`\`\`json ... \`\`\` tags. This can sometimes improve the reliability of parsing the JSON from the model's text response. For natively supported models, this parameter might add the wrapping around an already correct JSON string.

## Current Limitations

-   **No Simultaneous Tools and Structured Output**: Currently, you cannot use tools (as described in `tool_chat.md`) and request structured output by passing both `tools` and `structured_model` arguments in the **same** `query()` call. An attempt to do so will raise a `ValueError`.

-   **Sequential Use is Possible**: However, you *can* use tools and structured outputs sequentially. This is a powerful pattern where initial queries can use tools to fetch or compute data, and subsequent queries can process and structure this data using Pydantic models.

    Here's a more detailed biological example:

    **Scenario**: We want to find the ChEMBL ID for a drug, then get its mechanisms of action and putative targets, and finally structure this information.

    ```python
    import requests
    from biochatter.llm_connect import LangChainConversation
    from pydantic import BaseModel, Field
    from langchain_core.tools import tool
    from typing import List, Dict, Any, Optional
    import json
    from pprint import pprint

    # --- Step 1: Define the Tools ---

    @tool
    def get_chembl_id(drug_name: str) -> str:
        """
        Given a drug name, look up and return its ChEMBL ID using the ChEMBL API.
        Example: get_chembl_id(drug_name="aspirin")
        """
        url = "https://www.ebi.ac.uk/chembl/api/data/chembl_id_lookup/search.json"
        params = {"q": drug_name}
        headers = {"Accept": "application/json"}
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("chembl_id_lookups", [])
            if not hits:
                return f"Unable to find ChEMBL ID for {drug_name}"
            return hits[0].get("chembl_id", f"No ChEMBL ID found in hit for {drug_name}")
        except requests.RequestException as e:
            return f"Error querying ChEMBL API for {drug_name}: {str(e)}"
        except ValueError:
            return f"Invalid JSON received from ChEMBL API for {drug_name}"

    # GraphQL query for OpenTargets
    _GRAPHQL_QUERY = """
    query MechanismsOfActionSectionQuery($chemblId: String!) {
      drug(chemblId: $chemblId) {
        id
        mechanismsOfAction {
          rows {
            mechanismOfAction
            targetName
            targets {
              id
              approvedSymbol
            }
          }
        }
      }
    }
    """

    @tool
    def get_mechanisms_of_action(chembl_id: str) -> Dict[str, Any]:
        """
        Fetch mechanisms of action and target information for a given drug (by ChEMBL ID)
        using the OpenTargets GraphQL API.
        Example: get_mechanisms_of_action(chembl_id="CHEMBL25")
        """
        endpoint = "https://api.platform.opentargets.org/api/v4/graphql"
        payload = {"query": _GRAPHQL_QUERY, "variables": {"chemblId": chembl_id}}
        try:
            resp = requests.post(endpoint, json=payload, headers={"Content-Type": "application/json"}, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                return {"error": f"GraphQL errors for {chembl_id}: {data['errors']}"}
            return data.get("data", {})
        except requests.RequestException as e:
            return {"error": f"Error querying OpenTargets API for {chembl_id}: {str(e)}"}
        except ValueError:
            return {"error": f"Invalid JSON received from OpenTargets API for {chembl_id}"}

    # --- Step 2: Define the Pydantic Model for Structured Output ---

    class TargetDetail(BaseModel):
        approved_symbol: Optional[str] = Field(None, description="The approved symbol of the target (e.g., gene symbol).")
        target_id: Optional[str] = Field(None, description="The ID of the target (e.g., Ensembl ID).")

    class ActionMechanism(BaseModel):
        mechanism_of_action: Optional[str] = Field(None, description="Description of the mechanism of action.")
        target_name: Optional[str] = Field(None, description="The name of the target associated with this mechanism.")
        targets: List[TargetDetail] = Field(default_factory=list, description="List of specific targets involved in this mechanism.")

    class DrugTargetsOutput(BaseModel):
        drug_name_queried: str = Field(description="The original drug name queried.")
        chembl_id_found: Optional[str] = Field(None, description="The ChEMBL ID found for the drug.")
        mechanisms_and_targets: List[ActionMechanism] = Field(default_factory=list, description="List of mechanisms of action and associated targets.")
        error_message: Optional[str] = Field(None, description="Any error message encountered during the process.")

    # --- Step 3: Initialize Conversation (assuming already done) ---
    # convo = LangChainConversation(
    #     model_provider="google_genai", 
    #     model_name="gemini-2.0-flash",
    #     prompts={}
    # )
    # convo.set_api_key() # Ensure API key is set

    # --- Step 4: Sequential Queries ---

    # 4.1. Query to find ChEMBL ID
    drug_name = "imatinib"
    convo.query(f'Get the ChEMBL ID for the drug "{drug_name}"',tools=[get_chembl_id])

    # 4.2. Get the mechanisms of action and targets
    convo.query(f'Now get its mechanisms of action and targets',tools=[get_mechanisms_of_action])

    # 4.3. Get the structured output
    results =convo.query(
        "Return all the gathered information in a structured format", # Using the gene question from above
        structured_model=DrugTargetsOutput,
    )
    
    # --- Step 5: Parse the structured output ---
    # Convert the JSON string to a Python dictionary
    drug_data = json.loads(results[0])

    # Print the structured information
    print(f"Drug Name: {drug_data['drug_name_queried']}")
    print(f"ChEMBL ID: {drug_data['chembl_id_found']}")
    print("\nMechanisms of Action and Targets:")
    for i, mechanism in enumerate(drug_data['mechanisms_and_targets'], 1):
        print(f"\n{i}. Mechanism: {mechanism['mechanism_of_action']}")
        print(f"   Target: {mechanism['target_name']}")
        print("   Associated Targets:")
        for target in mechanism['targets']:
            print(f"     - {target['approved_symbol']} (ID: {target['target_id']})")

    if drug_data['error_message']:
        print(f"\nError: {drug_data['error_message']}")
    ```

    In this multi-step approach:
    1.  The first tool (`get_chembl_id`) is called to resolve a drug name to its ChEMBL ID.
    2.  The second tool (`get_mechanisms_of_action`) uses this ID to fetch detailed data from OpenTargets.
    3.  The final `query()` call doesn't use tools but instead provides the `DrugTargetsOutput` Pydantic model. The LLM is instructed to consolidate information from the preceding conversation turns (the initial drug name, the found ChEMBL ID, and the complex data from OpenTargets) into this predefined structure.

    This demonstrates how to chain tool calls and then use structured output to get a clean, predictable summary of the results.

This feature allows for more predictable and reliable interactions with LLMs when you need data in a specific format, streamlining integration with other parts of your application. 