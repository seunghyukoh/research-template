from accelerate.utils.memory import clear_device_cache, should_reduce_batch_size
from transformers.utils.logging import get_logger

logger = get_logger(__name__)


AUTO_BATCH_SIZE_TRAIN_STEPS = 3


def get_max_batch_size(func: callable, starting_batch_size=1) -> int:
    clear_device_cache(garbage_collection=True)

    def reduce_batch_size_fn():
        nonlocal batch_size
        batch_size = batch_size // 2
        return batch_size

    batch_size = starting_batch_size
    while True:
        if batch_size == 0:
            raise RuntimeError("No executable batch size found, reached zero.")
        try:
            logger.info(f"Trying batch size: {batch_size}")
            func(batch_size)  # 성공 여부만 확인
            return batch_size  # 성공한 batch_size 반환
        except Exception as e:
            if should_reduce_batch_size(e):
                clear_device_cache(garbage_collection=True)
                batch_size = reduce_batch_size_fn()
                logger.info(f"Reduced batch size to {batch_size}")
            else:
                raise


if __name__ == "__main__":
    import torch
    from transformers import AutoModelForCausalLM

    max_length = 1024
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", device_map="cuda")

    def example_func(batch_size):
        input_ids = torch.zeros((batch_size, max_length), dtype=torch.int64).cuda()
        labels = input_ids.clone()
        loss = model(input_ids, labels=labels).loss
        loss.backward()
        return batch_size

    batch_size = get_max_batch_size(example_func, starting_batch_size=32)
