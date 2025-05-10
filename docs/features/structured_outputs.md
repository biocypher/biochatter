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

    The process involves several steps:
    1.  Defining tools to fetch data from external APIs (ChEMBL and OpenTargets).
    2.  Defining Pydantic models to specify the desired structure for our final output.
    3.  Initializing the BioChatter conversation.
    4.  Making sequential queries: first to get the ChEMBL ID, then to get mechanisms of action, and finally to structure all gathered information.
    5.  Parsing and displaying the structured output.

    Let's look at the code for each step.

    **Step 1: Define the Tools**

    First, we define two tools. These tools will interact with external APIs to fetch the data we need.
    -   `get_chembl_id`: This tool takes a drug name and returns its ChEMBL ID.
    -   `get_mechanisms_of_action`: This tool takes a ChEMBL ID and retrieves the drug's mechanisms of action and target information from the OpenTargets API.

    ```python
    import requests
    from biochatter.llm_connect import LangChainConversation
    from pydantic import BaseModel, Field
    from langchain_core.tools import tool
    from typing import List, Dict, Any, Optional
    import json
    from pprint import pprint

    # Tool to get ChEMBL ID
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
        except ValueError: # If JSON parsing fails
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
        except ValueError: # If JSON parsing fails
            return {"error": f"Invalid JSON received from OpenTargets API for {chembl_id}"}
    ```
    **What we obtain:** We now have two functions, `get_chembl_id` and `get_mechanisms_of_action`, decorated with `@tool`. These are ready to be used by the BioChatter conversation object to perform their respective tasks.

    **Step 2: Define the Pydantic Model for Structured Output**

    Next, we define the Pydantic models that will determine the structure of our final output. This ensures the LLM returns the information in a consistent and parsable format.
    -   `TargetDetail`: Represents details of a biological target.
    -   `ActionMechanism`: Describes a mechanism of action, including its targets.
    -   `DrugTargetsOutput`: The main model that aggregates all information: the queried drug name, its ChEMBL ID, a list of its mechanisms and targets, and any potential error messages.

    ```python
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
    ```
    **What we obtain:** We have defined Python classes (`TargetDetail`, `ActionMechanism`, `DrugTargetsOutput`) that specify the exact schema we want our final data to conform to.

    **Step 3: Initialize Conversation**

    Now, we initialize the `LangChainConversation`. This object will manage the interaction with the LLM, including tool usage and structured output requests.

    ```python
    # Initialize the conversation
    convo = LangChainConversation(
        model_provider="google_genai", # Or any other supported provider
        model_name="gemini-2.0-flash", # Or a model suitable for your provider
        prompts={} # Using default prompts
    )
    convo.set_api_key() # Uncomment and set if your provider requires an API key directly
    ```
    **What we obtain:** A `convo` object is ready to interact with the specified LLM.

    **Step 4: Sequential Queries**

    This is where the core logic of the example unfolds. We make a series of calls to the `convo.query()` method:
    1.  **Find ChEMBL ID**: We ask the LLM to use the `get_chembl_id` tool to find the ChEMBL ID for the drug "imatinib".
    2.  **Get Mechanisms of Action**: Using the ChEMBL ID obtained (implicitly from the conversation history), we ask the LLM to use the `get_mechanisms_of_action` tool to fetch drug mechanisms and targets.
    3.  **Structure the Output**: Finally, we ask the LLM to consolidate all the information gathered in the previous steps into the `DrugTargetsOutput` structure we defined. No tools are passed in this step; only the `structured_model` is provided.

    ```python
    # 4.1. Query to find ChEMBL ID
    drug_name = "imatinib"
    query1_result = convo.query(
        f'Get the ChEMBL ID for the drug "{drug_name}"',
        tools=[get_chembl_id]
    )

    # 4.2. Get the mechanisms of action and targets
    # The LLM should use the ChEMBL ID from the previous turn's tool_result
    query2_result = convo.query(
        f'Now get its mechanisms of action and targets using the previously found ChEMBL ID.',
        tools=[get_mechanisms_of_action]
    )

    # 4.3. Get the structured output
    # The LLM will use the conversation history (drug name, ChEMBL ID, mechanism data)
    # to populate the DrugTargetsOutput model.
    results = convo.query(
        "Consolidate all the information gathered about imatinib, including its ChEMBL ID, "
        "and its mechanisms of action and targets, into the predefined structure. "
        structured_model=DrugTargetsOutput,
    )
    ```
    **What we obtain:**
    -   After the first query, the conversation history contains the ChEMBL ID for "imatinib" (e.g., "CHEMBL181").
    -   After the second query, the history includes detailed mechanism and target data from OpenTargets.
    -   After the third query, `structured_response_json` should hold a JSON string that conforms to our `DrugTargetsOutput` Pydantic model, containing all the consolidated information.

    **Step 5: Parse and Use the Structured Output**

    The final step is to parse the JSON string received from the LLM back into our Pydantic model. This allows us to easily access the data in a type-safe way.

    ```python
    # Parse the JSON string into the Pydantic model
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

    This multi-step approach demonstrates how to:
    1.  Use tools to gather information over several conversational turns.
    2.  Leverage structured outputs to consolidate and format the gathered information according to a predefined schema.

    **Complete Example Script**

    For convenience, here is the full script combining all the steps:

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
        except ValueError: # If JSON parsing fails
            return f"Invalid JSON received from ChEMBL API for {drug_name}"

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
        except ValueError: # If JSON parsing fails
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

    # --- Step 3: Initialize Conversation ---
    # Assuming LangChainConversation is already initialized and API key is set
    # For a self-contained script, you'd do:
    convo = LangChainConversation(
        model_provider="google_genai", # Replace with your provider e.g. "openai"
        model_name="gemini-2.0-flash",  # Replace with your model e.g. "gpt-4-turbo-preview"
        prompts={} # Using default prompts
    )
    
    # Set API key (read from environment variables)
    convo.set_api_key()

    # --- Step 4: Sequential Queries ---

    # 4.1. Query to find ChEMBL ID
    drug_name = "imatinib" # Example drug
    query1_result = convo.query(
        f'Get the ChEMBL ID for the drug "{drug_name}"',
        tools=[get_chembl_id]
    )

    # 4.2. Get the mechanisms of action and targets
    # The LLM should use the ChEMBL ID from the previous turn's tool_result
    query2_result = convo.query(
        f'Now get its mechanisms of action and targets using the previously found ChEMBL ID.',
        tools=[get_mechanisms_of_action]
    )

    # 4.3. Get the structured output
    # The LLM will use the conversation history (drug name, ChEMBL ID, mechanism data)
    # to populate the DrugTargetsOutput model.
    results = convo.query(
        "Consolidate all the information gathered about imatinib, including its ChEMBL ID, "
        "and its mechanisms of action and targets, into the predefined structure. ",
        structured_model=DrugTargetsOutput,
    )

    # --- Step 5: Parse and Use the Structured Output ---
    # Parse the JSON string into the Pydantic model
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