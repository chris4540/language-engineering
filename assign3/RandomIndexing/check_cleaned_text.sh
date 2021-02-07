#!/bin/bash
set -e

# remove the old example
rm -f cleaned_example.txt

# create the cleaned_example.txt
python random_indexing.py -c -co cleaned_example.txt

# check against the correct answer
diff -q correct_cleaned_example.txt cleaned_example.txt
if [ $? -ne 0 ]; then
    echo "The example is not the same to the correct one."
else
    echo "diff test passed."
fi
