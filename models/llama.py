import torch
from typing import List, Optional, Union
from lm_eval.base import BaseLM, CachingLM
from transformers import LlamaForCausalLM, LlamaTokenizer, BatchEncoding

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]


class LLaMA(BaseLM):
    def __init__(self, model_name, batch_size=1, device='cuda') -> None:
        self._batch_size = self.max_batch_size = batch_size
        self.seqlen = self._max_length = self._max_gen_toks = 2048
        self.add_special_tokens = False

        self.model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=False)
        self.model.eval()

        self._device = device
        self.config = self.model.config
        CachingLM(self, '__lmcache__')
        torch.set_grad_enabled(False)

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> Union[int, str, torch.device]:
        return self._device

    def tok_encode(self, string: str) -> TokenSequence:
        return self.tokenizer.encode(string, add_special_tokens=self.add_special_tokens)

    def tok_decode(self, tokens: torch.LongTensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_generate(self, context, max_length, eos_token_id):
        return None

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        return self.model(inputs)["logits"]
