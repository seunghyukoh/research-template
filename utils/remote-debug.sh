PORT=5678

function debug {
    WANDB_MODE=disabled python -m debugpy --listen 0.0.0.0:${PORT} --wait-for-client $@
}
