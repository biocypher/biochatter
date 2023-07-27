#!/bin/bash
set -ex

##
## Create some aliases
##
echo 'alias ll="ls -alF"' >> $HOME/.bashrc
echo 'alias la="ls -A"' >> $HOME/.bashrc
echo 'alias l="ls -CF"' >> $HOME/.bashrc

# Convenience workspace directory for later use
WORKSPACE_DIR=$(pwd)

# Change some Poetry settings to better deal with working in a container
pip install poetry==1.2.2
poetry config cache-dir ${WORKSPACE_DIR}/.cache
poetry config virtualenvs.create false

# Now install all dependencies
poetry install -E "streamlit podcast"