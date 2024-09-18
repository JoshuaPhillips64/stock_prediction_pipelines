#!/bin/bash

# Perform post-commit actions here, like notifications

# Example: Send a notification to Slack
curl -X POST -H 'Content-type: application/json' --data '{"text":"New commit pushed to the repository."}' https://hooks.slack.com/services/YOUR_SLACK_WEBHOOK_URL
