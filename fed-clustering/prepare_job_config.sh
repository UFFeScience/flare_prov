#!/usr/bin/env bash
DATASET_PATH="/tmp/nvflare/dataset/des.csv"

if [ ! -f "${DATASET_PATH}" ]
then
    echo "Please check if you saved Iris dataset in ${DATASET_PATH}"
fi

valid_frac=0.35
echo "Generating job configs with data splits, reading from ${DATASET_PATH}"

task_name="sklearn_kmeans"
for site_num in 3;
do
    python3 utils/prepare_job_config.py \
    --task_name "${task_name}" \
    --data_path "${DATASET_PATH}" \
    --site_num ${site_num} \
    --valid_frac ${valid_frac}
done