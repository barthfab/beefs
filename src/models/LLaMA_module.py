from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import Any, List
import pytorch_lightning as pl
from src.utils.eval_script import a2_evaluation, local_evaluation
from src.utils.example_creators import down_sample_prompt

class LLaMA(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        self.model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
        self.local_eval = kwargs['local_eval']
        self.do_sampling = kwargs['do_sampling']
        self.top_k = kwargs['top_k']
        self.top_p = kwargs['top_p']
        self.temperature = kwargs['temperature']
        try:
            self.output = kwargs['output']
        except:
            self.output = 'logs/train_result'
        try:
            self.arg_finder = kwargs['arg_finder']
        except:
            self.arg_finder = 0

    def step(self, batch: Any):
        prompt, example = batch
        out_len = self.tokenizer(example.output_tokens, return_tensors="pt").input_ids.size(dim=1)
        out_len = out_len + 15
        inputs = self.tokenizer(prompt, return_tensors="pt")
        i = 0
        while inputs.input_ids.size(dim=1) + out_len >= 4096:
            i += 1
            prompt = down_sample_prompt(example, prompt)
            inputs = self.tokenizer(prompt, return_tensors="pt")
        if i > 0:
            print(f'WARNING: token length: {i} examples are removed from the prompt')
        output = self.forward(inputs, out_len)
        output_prompt = output.split(prompt)[-1].split('\n')[0]
        return output_prompt, prompt, example

    def forward(self, inputs, out_len):
        inputs = inputs.to(self.device)
        generate_ids = self.model.generate(inputs.input_ids,
                                           attention_mask=inputs.attention_mask,
                                           max_length=inputs.input_ids.size(dim=1) + out_len,)
        """
        top_k=self.top_k,
        temperature=self.temperature,
        top_p=self.top_p,"""
        output = self.tokenizer.batch_decode(generate_ids,
                                             attention_mask=inputs.attention_mask,
                                             skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)[0]
        return output

    def training_step(self, batch: Any, batch_idx: int):
        prompt_choices, input_prompt, example = self.step(batch)
        return {"output_prompt": prompt_choices, "input_prompt": input_prompt, "example": example}

    def validation_step(self, batch: Any, batch_idx: int):
        prompt_choices, input_prompt, example = self.step(batch)
        return {"output_prompt": prompt_choices, "input_prompt": input_prompt, "example": example}

    def test_step(self, batch: Any, batch_idx: int):
        prompt_choices, input_prompt, example = self.step(batch)
        return {"output_prompt": prompt_choices, "input_prompt": input_prompt, "example": example}

    def test_epoch_end(self, outputs: List[Any]):
        if self.local_eval:
            f1, prec, rec = local_evaluation(outputs, self.arg_finder)
        else:
            f1, prec, rec = a2_evaluation(outputs, self.output, self.arg_finder)
        self.log("val/f1", f1, on_epoch=True)
        self.log("val/precision", prec, on_epoch=True)
        self.log("val/recall", rec, on_epoch=True)

    def configure_optimizers(self):
        return None

    def training_epoch_end(self, outputs: List[Any]):
        return -1

    def validation_epoch_end(self, outputs: List[Any]):
        return -1


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
