#!/usr/bin/env bash

ENVIRONMENT_NAME="qch"
source "$(conda info --base)/etc/profile.d/conda.sh"

conda env list | grep -q "^$ENVIRONMENT_NAME\s" || conda create --name $ENVIRONMENT_NAME python=3.12 -y
conda activate $ENVIRONMENT_NAME

if [ -f "environmnet.yml" ]; then
    conda env update --file environment.yml --prune -y
fi

pip install -e .

