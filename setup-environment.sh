source .env

ENV_NAME=$PROJECT_NAME
CONDA_PATH=${CONDA_ROOT}/bin/conda
ENV_PATH=${CONDA_ROOT}/envs/${ENV_NAME}/bin/python
PIP_PATH=${CONDA_ROOT}/envs/${ENV_NAME}/bin/pip

if [ -e ${ENV_PATH} ]; then
    echo "env directory already exists"
else
    conda create -n ${ENV_NAME} python=3.12 -y
fi

${PIP_PATH} install -q torch==2.5.1
${PIP_PATH} install -q -r requirements.txt

${CONDA_ROOT}/envs/${ENV_NAME}/bin/pre-commit install
