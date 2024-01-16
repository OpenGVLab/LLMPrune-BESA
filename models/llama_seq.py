import torch
from typing import List, Optional, Union
from lm_eval.base import BaseLM, CachingLM
from transformers import LlamaForCausalLM, LlamaTokenizer, BatchEncoding

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]


class LLaMA_Seq(BaseLM):
    def __init__(self, model_name, batch_size=1, device='cuda') -> None:
        self._batch_size = self.max_batch_size = batch_size
        self.seqlen = self._max_length = self._max_gen_toks = 2048
        self.add_special_tokens = False
        
        self.model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cpu")
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
        self.model.model.embed_tokens = self.model.model.embed_tokens.to(self.device)
        layers = self.model.model.layers
        layers[0] = layers[0].to(self.device)

        cache = {'hidden_states': None, 'attention_mask': None, 'position_ids': None}
        class Catcher(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                cache['hidden_states'] = inp
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_ids'] = kwargs['position_ids']
                raise ValueError

        layers[0] = Catcher(layers[0])
        try:
            self.model(inputs)
        except ValueError:
            pass
        layers[0] = layers[0].module
        layers[0] = layers[0].cpu()
        self.model.model.embed_tokens = self.model.model.embed_tokens.cpu()
        torch.cuda.empty_cache()

        inps = cache.pop('hidden_states')
        outs = torch.zeros_like(inps)
        for i in range(len(layers)):
            layer = layers[i].to(self.device)
            outs = layer(inps, **cache)[0]
            layers[i] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
            inps, outs = outs, inps

        self.model.model.norm = self.model.model.norm.to(self.device)
        self.model.lm_head = self.model.lm_head.to(self.device)
        inps = self.model.model.norm(inps)
        lm_logits = self.model.lm_head(inps)
        self.model.model.norm = self.model.model.norm.cpu()
        self.model.lm_head = self.model.lm_head.cpu()
        torch.cuda.empty_cache()

        return lm_logits
