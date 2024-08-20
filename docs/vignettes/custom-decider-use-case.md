# Custom BioChatter Light and Next: Cancer Genetics Use Case

This example is part of the BioChatter manuscript supplement.

<!-- TODO DOI -->

## Background

Personalized medicine tailors treatment to a patient's unique genetic makeup.
In cancer care, this approach helps categorize patients and assign them to specific treatment groups in clinical trials.
However, interpreting and making decisions based on this data is challenging due to the complexity of genetic variations, the interaction between genes and environmental factors, tumor diversity, patient histories, and the vast amount of data produced by advanced technologies.

In the [DECIDER consortium](https://deciderproject.eu), we aim to improve clinical decisions by providing support systems, for instance for the geneticists working on these cases.
The code for the use case lives at [https://github.com/biocypher/decider-genetics](https://github.com/biocypher/decider-genetics).

Below, we show how we build a support application for this use case.

## Sources of knowledge

We integrate knowledge from diverse resources, using [BioCypher](https://biocypher.org) to build a knowledge graph of:

1. Processed whole genome sequencing data of ovarian cancer patients (synthetic data)

    - genomic changes

        - classified by consequence (protein truncation, amino acid change)

        - algorithmic prediction of deleteriousness

        - variant identifiers 

    - allele dosages

        - gene allele copy number (amplifications, deletions, loss-of-heterogeneity)

        - mutation pervasiveness (estimate of number of affected alleles, or suspected subclonality)
        
    - proportion of cancer cells in the sample (tumour purity)
        

2. the patients' clinical history (synthetic data)

    - personal information (age at diagnosis, BMI, etc.)

    - treatment history, known side effects, clinical response

    - lab test results (blood, imaging, histopathology)

    - common treatment-relevant mutations (BRCA), HR deficiency, PARP-inhibitor maintenance

3. data from open resources (real data)

    - variant annotations (as provided by the genetics pipeline of the DECIDER consortium)

    - gene annotations (as provided by the genetics pipeline of the DECIDER consortium)

    - pathway / process annotations (from public databases such as [Gene Ontology](http://geneontology.org))

    - drug annotations (from [OncoKB](https://www.oncokb.org))

In addition, we provide access to more resources via the RAG and API agents:

1. relevant publications from
[PubMed](https://pubmed.ncbi.nlm.nih.gov/?term=high%20grade%20serous%20ovarian%20cancer&filter=simsearch2.ffrft&filter=pubt.review&filter=pubt.systematicreview)
(real data) embedded in a vector database

2. relevant knowledge streamed live from OncoKB (see below) via API access through BioChatter's API agent

## The geneticist's workflow

Personalized cancer therapy is guided by identifying somatic genomic driver events in specific genes, particularly when these involve well-known hotspot mutations. However, unique somatic events in the same genes or pathways can create a "grey zone" that requires manual geneticist analysis to determine their clinical significance.

To address this, a comprehensive BioCypher backend processes whole-genome sequencing data to catalog somatic changes, annotating their consequences and potential actionability.
These data can then be linked to external resources for clinical interpretation.
For example, certain mutations in the BRCA1 or ERBB2 genes can indicate sensitivity to specific treatments like PARP inhibitors or trastuzumab.

To fully leverage actionable data, the integration of patient-specific information with literature on drug targets and mechanisms of action or resistance is essential. [OncoKB](https://www.oncokb.org/actionable-genes#sections=Tx) is the primary resource for this information, accessible through drug annotations added to the knowledge graph (KG) and via the BioChatter API calling mechanism.

Additionally, semantic search tools facilitate access to relevant biomedical literature, enabling geneticists to quickly verify findings against established treatments or resistance mechanisms.

In summary, the main contributions of our use case to the productivity of this workflow are:

- making processed and analysed genomic data locally available in a centralised resource by building a custom KG

- allowing comparison to literature via semantic search inside a vector database with relevant publications

- providing live access to external resources via the API agent

<!-- OncoKB annotated - drug, cancer, resistance

TODO add some to welcome page

Questions:
# meta level
How many patients do we have on record?
what was patient1's response to previous treatment, and which treatment did they receive?
which patients have hr deficiency but have not received parp inhibitors?
how many patients had severe adverse reactions, and to which drugs

# genetics
Does patient1 have a sequence variant in a gene that is druggable? Which drug, and what evidence level has the association?
Does patient1 have a sequence variant in a gene that is druggable with evidence level "1"? Which drug? Return unique values.
Does patient1 have a copy number variant in a gene that is druggable with evidence level "1"? Which drug? Return unique values.
Does patient1 have a sequence variant in a gene that is druggable with evidence level "1"? Which drug? Return unique values and the variant information for each drug. Only select variants with CADD_phred above 5.
What is the variant with the highest CADD_phred of the samples of the patient with id "patient1"
How many clinically significant (CLNSIG = Pathogenic) variants does each patient have
- used to distinguish BRCA mutations (there are benign ones, so don't benefit from PARP-I)
How many clinically significant (CLNSIG = Pathogenic or Likely_pathogenic) variants does each patient have
How many variants of unclear clinical significance (CLNSIG = Uncertain_significance or Conflicting_interpretations_of_pathogenicity) does each patient have
which clinically significant (CLNSIG = Pathogenic) sequence variants do the samples of patient5 have?
which patients have sequence and copy number variants in the same gene?
What is the sequence variant with the highest CADD_phred, and which patient has it
which copy number alterations are exclusive to patient1
is there a patient with overlapping variants compared to patient1

# biology
what are the biological functions of the gene SETBP1 (??)

Non-funtional:
which genes of patient2 have more than 2 nMajor copies

Taru - Geneticist: ideal to have all data and evidence in the same place; if it’s easy case, make already a recommendation, give standard interpretation.

Create prompt with explanation of the thought process and important parameters regarding the variants etc? -->

## Building the application

We will explain how to use the BioCypher ecosystem, specifically, BioCypher and BioChatter, to build a decision support application for a cancer geneticist.
The code base for this use case, including all details on how to set up the KG and the applications, is available at [https://github.com/biocypher/decider-genetics](https://github.com/biocypher/decider-genetics).
You can find live demonstrations of the application at links provided in the README of the repository.
The build procedures can be reproduced by cloning the repository and running `docker-compose up -d` (or the equivalent for the Next app) in the root directory (note that the default configuration requires authentication with OpenAI services).
The process involves the following steps:

1. Identifying data sources and creating a knowledge graph schema

2. Building the KG with BioCypher from the identified sources

3. Using BioChatter Light to develop and troubleshoot the KG application

4. Customising BioChatter Next to yield an integrated conversational interface

5. Deploying the applications

### Identifying data sources and creating a knowledge graph schema

We examine the data sources described above and design a KG schema that can accommodate the data.
The configuration file, [schema_config.yaml](https://github.com/biocypher/decider-genetics/blob/main/config/schema_config.yaml), can be seen in the `config` directory of the repository.
The schema should also be designed with LLM access in mind; performance in generating specific queries can be adjusted for in step three (troubleshooting using BioChatter Light).
We created a bespoke adapters for the genetics data of the DECIDER cohort according to the output format of the genetics pipeline, and reused existing adapters for the open resources.
They can be found in the [decider_genetics/adapters](https://github.com/biocypher/decider-genetics/tree/main/decider_genetics/adapters) directory of the repository.
For this use case, we created synthetic data to stand in for the real data for privacy reasons; the synthetic data are available in the `data` directory.

### Building the KG with BioCypher

In the dedicated adapters for the DECIDER genetics data, we pull the data from the synthetic data files and build the KG.
We perform simplifying computations, as described above, to facilitate standard workflows (such as counting alleles, identifying pathogenic variants, and calculating tumour purity).
We mold the data into the specified schema in a transparent and reproducible manner by configuring the adapters (see the [decider_genetics/adapters](https://github.com/biocypher/decider-genetics/tree/main/decider_genetics/adapters) directory).

After creating the schema and adapters, we run the build script to populate the KG.
BioCypher is configured using the [biocypher_config.yaml](https://github.com/biocypher/decider-genetics/blob/main/config/biocypher_config.yaml) file in the `config` directory.
Using the Docker Compose workflow included in the BioCypher template repository, we build a containerised version of the KG.
We can inspect the KG in the Neo4j browser at `http://localhost:7474` after running the build script.
Any changes, if needed, can be made to the configuration of schema and adapters.

### Using BioChatter Light to develop and troubleshoot the KG application

Upon deploying the KG via Docker, we can use a custom BioChatter Light application to interact with the KG.
Briefly, we remove all components except the KG interaction panel via environment variables in the [docker-compose.yml](https://github.com/biocypher/decider-genetics/blob/main/docker-compose.yml) file (see also the corresponding [vignette](custom-bclight-simple.md)).
This allows us to start the KG and interact with it using an LLM in a reproducible manner with just one command.
We can then test the LLM-KG interaction by asking questions and examining the generated queries and its results from the KG.
Once we are satisfied with the KG schema and LLM performance, we can advance to the next step.

The BioChatter Light application, including the KG creation, can be built using `docker compose up -d` in the root directory of the [repository](https://github.com/biocypher/decider-genetics).
An online demonstration of this application can be found at []().

<!-- TODO show online version -->

### Customising BioChatter Next to yield an integrated conversational interface

We can further customise the Docker workflow to start the BioChatter Next application, including its REST API middleware `biochatter-server`.
In addition to deploying all software components, we can also customise its appearance and functionality.
Using the [biochatter-next.yaml](https://github.com/biocypher/decider-genetics/blob/main/config/biochatter-next.yaml) configuration file (in `config`, as all other configuration files), we can adjust the welcome message, how-to-use section, the system prompts for the LLM, which tools can be used by the LLM agent, the connection details of externally hosted KG or vectorstore, and other parameters.
We then start BioChatter Next using a [dedicated Docker Compose file](https://github.com/biocypher/decider-genetics/blob/main/docker-compose-next.yml), which includes the `biochatter-server` middleware and the BioChatter Next application.

The BioChatter Next application, including the customisation of the LLM and the integration of the KG, can be built using `docker compose -f docker-compose-next.yml up -d` in the root directory of the [repository](https://github.com/biocypher/decider-genetics).
An online demonstration of this application can be found at []().

<!-- TODO show online version -->

### Deploying the applications

The final step is to deploy one or both applications on a server.
Using the Docker Compose workflow, we can deploy the applications in many different environments, from local servers to cloud-based solutions.
The environment supplied by the Docker software allows for high reproducibility and easy scaling.
The BioChatter Light app can be used for testing, but also to provide a simple one-way interface to the KG for users who do not need the full conversational interface.
The BioChatter Next app can be configured to connect to KG and vectorstore deployments on different servers, allowing for a distributed architecture and dedicated maintenance of components; but it can also be deployed in tandem from one Docker Compose, for smaller setups or local use.

