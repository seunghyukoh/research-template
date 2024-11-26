from transformers import Trainer as HFTrainer
from .loss_functions import LOSS_FUNCTION_FACTORIES


class Trainer(HFTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        loss_function_factory = LOSS_FUNCTION_FACTORIES[self.args.loss_type]
        self.loss_function = loss_function_factory(self)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # Custom loss for training
        custom_losses = self.loss_function(
            model=model,
            inputs=inputs,
            return_outputs=return_outputs,
        )

        return custom_losses
