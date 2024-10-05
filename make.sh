#!/bin/bash
if ! python3 --version > /dev/null 2>&1; then
    echo "Error: python3 is not installed. Please install python3."
    exit 1
fi

echo "project setup completed sucessfully."