def factory(trainer):
    def loss_function(model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss

        if return_outputs:
            return (loss, outputs)

        return loss

    return loss_function
