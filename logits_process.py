# this code snippet is taken from the Huggingface implementation of Min P

import inspect
import math
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from ..utils import add_start_docstrings
from ..utils.logging import get_logger


logger = get_logger(__name__)

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""


class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsProcessorList(list):
    """
    This class can be used to create a list of [`LogitsProcessor`] or [`LogitsWarper`] to subsequently process a
    `scores` input tensor. This class inherits from list and adds a specific *__call__* method to apply each
    [`LogitsProcessor`] or [`LogitsWarper`] to the inputs.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            kwargs (`Dict[str, Any]`, *optional*):
                Additional kwargs that are specific to a logits processor.

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.

        """
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)

        return scores

class MinPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs min-p, i.e. keeps all tokens that are above a minimum probability, scaled by the
    probability of the most likely token. As a result, the filter becomes more agressive in the presence of
    high-probability tokens, which is a sign of a confident output that we shouldn't deviate from.

    Often used together with [`TemperatureLogitsWarper`]. Used as an alternative to [`TopPLogitsWarper`] and
    [`TopKLogitsWarper`].

    Created by [REDACTED]. Code adapted from [REDACTED].

    Args:
        min_p (`float`):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between 0 and 1, with typical values in the 0.01-0.2 range.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(1)
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3 | < 4 (left-hand pointer) ;
    <BLANKLINE>
    <BLANKLINE>

    >>> # With `min_p` sampling, the output gets restricted to high-probability tokens.
    >>> # Pro tip: In practice, LLMs use `min_p` in the 0.01-0.2 range.
    >>> outputs = model.generate(**inputs, do_sample=True, min_p=0.1)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
    """

    def __init__(self, min_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if min_p < 0 or min_p > 1.0:
            raise ValueError(f"`min_p` has to be a float >= 0 and <= 1, but is {min_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.min_p = min_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Convert logits to probabilities
        probs = torch.softmax(scores, dim=-1)
        # Get the probability of the top token for each sequence in the batch
        top_probs, _ = probs.max(dim=-1, keepdim=True)
        # Calculate the actual min_p threshold by scaling min_p with the top token's probability
        scaled_min_p = self.min_p * top_probs
        # Create a mask for tokens that have a probability less than the scaled min_p
        tokens_to_remove = probs < scaled_min_p

        sorted_indices = torch.argsort(scores, descending=True, dim=-1)
        sorted_indices_to_remove = torch.gather(tokens_to_remove, dim=-1, index=sorted_indices)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., : self.min_tokens_to_keep] = False

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed
