from .cross_entropy import factory as cross_entropy_loss

LOSS_FUNCTION_FACTORIES = {
    "cross_entropy": cross_entropy_loss,
}
