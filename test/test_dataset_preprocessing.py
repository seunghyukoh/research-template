"""Unit tests for dataset preprocessing functions."""

import pytest

from utils.dataset_preprocessing import (
    get_preprocessing_fn,
    list_preprocessing_fns,
    register_preprocessing_fn,
)


class TestPreprocessingRegistry:
    """Tests for the preprocessing function registry."""

    def test_list_preprocessing_fns(self):
        """Test listing all preprocessing functions."""
        fns = list_preprocessing_fns()
        assert isinstance(fns, list)
        assert "gsm8k" in fns
        assert fns == sorted(fns)  # Should be sorted

    def test_get_existing_function(self):
        """Test retrieving an existing preprocessing function."""
        fn = get_preprocessing_fn("gsm8k")
        assert callable(fn)

    def test_get_nonexistent_function(self):
        """Test that getting a nonexistent function raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_preprocessing_fn("nonexistent_function")

        error_msg = str(exc_info.value)
        assert "not found" in error_msg.lower()
        assert "available functions" in error_msg.lower()

    def test_register_duplicate_function(self):
        """Test that registering a duplicate function raises ValueError."""
        # gsm8k is already registered
        with pytest.raises(ValueError) as exc_info:

            @register_preprocessing_fn("gsm8k")
            def duplicate_fn(example):
                return example

        error_msg = str(exc_info.value)
        assert "already registered" in error_msg.lower()


class TestGSM8KPreprocessing:
    """Tests for GSM8K preprocessing function."""

    def test_basic_preprocessing(self):
        """Test basic GSM8K preprocessing."""
        fn = get_preprocessing_fn("gsm8k")

        example = {
            "question": "What is 2+2?",
            "answer": "The answer is 4.",
        }

        result = fn(example)

        assert "prompt" in result
        assert "completion" in result
        assert result["prompt"] == "What is 2+2?"
        assert result["completion"] == "The answer is 4."

    def test_preserves_original_fields(self):
        """Test that preprocessing doesn't modify original fields."""
        fn = get_preprocessing_fn("gsm8k")

        example = {
            "question": "What is 2+2?",
            "answer": "4",
            "extra_field": "should be preserved",
        }

        result = fn(example)

        # Check that only prompt and completion are in result
        assert set(result.keys()) == {"prompt", "completion"}

    def test_empty_strings(self):
        """Test handling of empty strings."""
        fn = get_preprocessing_fn("gsm8k")

        example = {
            "question": "",
            "answer": "",
        }

        result = fn(example)

        assert result["prompt"] == ""
        assert result["completion"] == ""

    def test_missing_question_field(self):
        """Test that missing question field raises KeyError."""
        fn = get_preprocessing_fn("gsm8k")

        example = {
            "answer": "Test answer",
        }

        with pytest.raises(KeyError) as exc_info:
            fn(example)

        error_msg = str(exc_info.value)
        assert "question" in error_msg.lower()

    def test_missing_answer_field(self):
        """Test that missing answer field raises KeyError."""
        fn = get_preprocessing_fn("gsm8k")

        example = {
            "question": "Test question",
        }

        with pytest.raises(KeyError) as exc_info:
            fn(example)

        error_msg = str(exc_info.value)
        assert "answer" in error_msg.lower()

    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored."""
        fn = get_preprocessing_fn("gsm8k")

        example = {
            "question": "Test question",
            "answer": "Test answer",
            "extra": "should be ignored",
        }

        result = fn(example)

        assert set(result.keys()) == {"prompt", "completion"}


class TestCustomRegistration:
    """Tests for custom preprocessing function registration."""

    def test_register_and_use_custom_function(self):
        """Test registering and using a custom preprocessing function."""

        @register_preprocessing_fn("test_custom_func")
        def preprocess_custom(example):
            return {
                "prompt": f"Custom: {example['input']}",
                "completion": f"Output: {example['output']}",
            }

        # Check it's registered
        assert "test_custom_func" in list_preprocessing_fns()

        # Check it works
        fn = get_preprocessing_fn("test_custom_func")
        result = fn({"input": "hello", "output": "world"})

        assert result["prompt"] == "Custom: hello"
        assert result["completion"] == "Output: world"

    def test_decorator_returns_function(self):
        """Test that the decorator returns the original function."""

        @register_preprocessing_fn("test_decorator_return")
        def my_fn(example):
            return {"prompt": "p", "completion": "c"}

        # The decorator should return the original function
        assert callable(my_fn)
        assert my_fn.__name__ == "my_fn"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
