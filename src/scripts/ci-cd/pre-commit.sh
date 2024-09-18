#!/bin/bash

# Run linters and tests here

# Example: Run pylint on Python files
find . -name "*.py" -not -path "./venv/*" -exec pylint {} \;
