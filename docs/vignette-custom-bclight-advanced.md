# Custom BioChatter Light: Advanced Use Case

For more advanced workflows, you may want to customise the BioChatter Light
interface to display a new way of visualising your data or interacting with it.
Due to the flexible, modular, and easy-to-use [Streamlit](https://streamlit.io)
framework, you can easily create new tabs and customise the existing ones. Here,
we show how to create new tabs on the example of a project management tool we
developed as a demonstration use case. The final web app is available at
[https://project.biochatter.org](https://project.biochatter.org).

## Background

Managing a scientific group is challing for multiple reasons, particularly one
which has multiple interdependent projects, each of which is pursued by a small
team of junior and senior researchers. To enhance productivity and
communication, it could be beneficial to have a tool that takes away some of the
burden of project management. In the context of our work, there are two
components we see as essential: data management (FAIR and transparent) and
simple interfaces (driven by conversational AI). Naturally, we will be using
BioCypher and BioChatter for these two components.

We will use a GitHub Project board
([here](https://github.com/orgs/biocypher/projects/6/views/1)) as the "ground
truth" for our project management tool. This is close to a real-world scenario
and allows connectivity to code repositories, issues, pull requests, and other
components of computational collaboration. The linked project is "synthetic"
data for demonstration purposes. The repository to build the KG and deploy the
BioChatter Light web app is available
[here](https://github.com/biocypher/project-planning).

## Build the KG

We modified an existing adapter for the GitHub GraphQL API to pull data from the
GitHub Project board. Thus, the time investment to build the KG was minimal
(~3h); this is one central principle of BioCypher. We adapted the code
(`project_planning/adapters/github_adapter.py`) and KG schema
(`config/schema_config.yaml`) to represent the relevant features of the GitHub
Project board. The pre-existing KG build and deploy scripts were used via the
`docker-compose.yml` file. For public deployment, we also added a
`docker-compose-password.yml`, which builds a password-protected version of the
KG. Deployment and setup of the cloud VM took another ~2h.

Be aware that running this script will require a GitHub token with access to the
project board. This token should be stored in the environment variable
`BIOCYPHER_GITHUB_PROJECT_TOKEN`.

## Add the additional tabs to BioChatter Light

BioChatter Light has a modular architecture to accommodate flexible layout
changes. We also added a configuration that allows turning on or off specific
tabs via environment variables. For this project, we added three new tabs:
"Summary", "Tasks", and "Settings". The "Summary" tab shows an overview of the
completed tasks in the current iteration of the project, the "Tasks" tab shows
the upcoming tasks of the group and each team member, and the "Settings" tab
allows configuration of the queries and LLM instructions used to generate the
content for the other tabs.

The solution as a web app is not the ideal use case for the project management
tool; rather, we envision the deployed version as an integration of common
messengers (Zulip, Slack, etc.) that acts as a conversational assistant to the
group and its members. The web app is a proof of concept and demonstration of
the capabilities of BioChatter Light, simulating feedback to the group and
individual users via the simplified interface.

The tabs were added to the BioChatter Light codebase in the corresponding module
(`components/panels/project.py`), which contains all three tabs. The Streamlit
framework makes this relatively easy; each tab only requires about 100 lines of
code in this module and only contains simple components such as columns,
buttons, and text fields. We also added environment variables to the
configuration (`components/config.py`) to allow turning on or off the new tabs.

## Configure the BioChatter Light Docker container

As in the previous vignette, we can now configure the BioChatter Light Docker
container to show only the new tabs. We provide these settings via the
environment variables we introduced above, while turning off the default tabs.
We have also added configurable environment variables for setting a page title,
header, and subheader for the web app without having to change the source code.

```yaml
services:
  ## ... build, import, and deploy the KG ...
  app:
    image: biocypher/biochatter-light:0.6.10
    container_name: app
    ports:
      - "8501:8501"
    networks:
      - biochatter
    depends_on:
      import:
        condition: service_completed_successfully
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - BIOCHATTER_LIGHT_TITLE=Project Planner
      - BIOCHATTER_LIGHT_HEADER=GitHub Project Planning Assistant
      - BIOCHATTER_LIGHT_SUBHEADER=A BioChatter Demonstration App for integrated project planning using LLMs
      - DOCKER_COMPOSE=true
      - CHAT_TAB=false
      - PROMPT_ENGINEERING_TAB=false
      - RAG_TAB=false
      - CORRECTING_AGENT_TAB=false
      - KNOWLEDGE_GRAPH_TAB=false
      - LAST_WEEKS_SUMMARY_TAB=true
      - THIS_WEEKS_TASKS_TAB=true
      - TASK_SETTINGS_PANEL_TAB=true
```

You can see the full configuration in the `docker-compose.yml` file of the
[project-planning](https://github.com/biocypher/project-planning) repository.
For public deployment, we also added a password-protected version of the KG,
which only requires a few additional lines in the `docker-compose-password.yml`
file. To deploy the tool on a cloud VM, we now only need to run the following
commands:

```bash
git clone https://github.com/biocypher/project-planning.git
docker-compose -f project-planning/docker-compose-password.yml up -d
```

We just need to make sure to provide an `OPENAI_API_KEY` and a
`BIOCYPHER_GITHUB_PROJECT_TOKEN` in the VM's environment to be accessed by the
Docker workflow.

## Useful tips for deployment

Many vendors offer cloud VMs with pre-installed Docker and Docker Compose, as
well as Nginx for reverse proxying. We recommend using a reverse proxy to
provide HTTPS and a domain name for the web app. This can be done with a few
lines in the Nginx configuration file. For example, to deploy the project
management tool on a cloud VM with a domain name `project.biochatter.org`, you
can use the following Nginx configuration:

```nginx
server {
    listen 80;
    server_name project.biochatter.org;

    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

You can find more info
[here](https://www.digitalocean.com/community/tutorials/how-to-configure-nginx-as-a-reverse-proxy-on-ubuntu-22-04).
We also recommend to set up certification with Let's Encrypt for HTTPS. This can
be done with the Certbot tool, which is available for most Linux distributions.

In addition, you need to make sure that your Neo4j deployment is accessible from
your web app, and that the connection is secure. You can either make the DB
accessible only on the VM's network, which would allow running it without
encryption, or you can set up a secure connection with a password. Both options
are implemented in the `docker-compose.yml` and `docker-compose-password.yml`
files of the `project-planning` repository. You can find more info
[here](https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-neo4j-on-ubuntu-20-04).