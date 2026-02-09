#!/bin/bash

# set the evaluation files
evaluate_full=false
evaluate_clip=true


RESULT_FOLDER="/path/to/your/model_test_result_folder"

if [ "$evaluate_full" = true ]; then
    echo "Running evaluate_metrics_full.py..."
    python /mnt/data2/wgk/surgvidlm_project/surgvidlm_open/evaluation/evaluate_metrics_full.py --model_test_result_folder "$RESULT_FOLDER"
fi

if [ "$evaluate_clip" = true ]; then
    echo "Running evaluate_metrics_clip.py..."
    python /mnt/data2/wgk/surgvidlm_project/surgvidlm_open/evaluation/evaluate_metrics_clip.py --model_test_result_folder "$RESULT_FOLDER"
fi
