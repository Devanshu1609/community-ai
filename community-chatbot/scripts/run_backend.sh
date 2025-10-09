#!/bin/bash

python scripts/github_agent.py &

python scripts/jira_agent.py &

python scripts/slack.py &

echo "All servers started!"