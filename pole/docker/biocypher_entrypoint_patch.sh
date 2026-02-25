#!/bin/bash -eu

if [ $FILL_DB_ON_STARTUP == 'yes' ]; then
  bash import.sh
  bash create_table.sh &
fi
