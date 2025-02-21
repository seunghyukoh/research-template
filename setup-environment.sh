UV_PATH="${HOME}/.local/bin/uv"

if [ ! -f ${UV_PATH} ]; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

${UV_PATH} sync
