#!/bin/bash

# Check the argument passed to the container
case "$1" in
    parseData)
        # Assuming your XML parsing script is named xml_parsing_script.py and located at /app
        shift  # Shift script arguments (removes the first one, which is 'parseData')
        python ./dataParser/xmldataparser.py "$@"  # Pass remaining arguments to the script
        ;;
    healthBot)
        # If your health bot script is appleHealthBot.py and located at /app
        python ./healthBot/appleHealthBot.py
        ;;
    *)
        echo "Usage: $0 {parseData|healthBot} [args for parseData]"
        exit 1
        ;;
esac
