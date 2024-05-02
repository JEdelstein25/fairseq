#!/bin/bash

    # Extract text data from CSV (assuming the text is in the 'text' column and skipping the header if there is one)
#awk -F',' 'NR > 1 {print $10}' gpt2_bpe/weather-agg-DFE.csv > gpt2_bpe/weather-agg-DFE.txt

# Run BPE encoding on the extracted text
fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref gpt2_bpe/weather-agg-DFE.bpe \
    --validpref gpt2_bpe/weather-agg-DFE.bpe \
    --testpref gpt2_bpe/weather-agg-DFE.bpe \
    --destdir gpt2_bpe/weather_agg \
    --workers 60
