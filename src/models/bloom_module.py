from transformers import BloomTokenizerFast
from typing import Any, List
import pytorch_lightning as pl
from src.utils.eval_script import a2_evaluation, local_evaluation
from petals.client import DistributedBloomForCausalLM
from src.utils.example_creators import down_sample_prompt


class Bloom(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.tokenizer = BloomTokenizerFast.from_pretrained(kwargs['model'], padding_side='right')
        model = DistributedBloomForCausalLM.from_pretrained(kwargs['model'])
        self.model = model.cuda()
        self.do_sampling = kwargs['do_sampling']
        self.top_k = kwargs['top_k']
        self.temperature = kwargs['temperature']
        self.num_beams = kwargs['num_beams']
        self.early_stopping = kwargs['early_stopping']
        self.num_return_sequences = kwargs['num_return_sequences']
        try:
            self.output = kwargs['output']
        except:
            self.output = 'logs/train_result'
        try:
            self.arg_finder = kwargs['arg_finder']
        except:
            self.arg_finder = 0
        try:
            self.local_eval = kwargs['local_eval']
        except:
            self.local_eval = True

    def forward(self, inputs, out_len):
        inputs = inputs.to('cuda')
        generate_ids = self.model.generate(inputs.input_ids,
                                           attention_mask=inputs.attention_mask,
                                           max_length=inputs.input_ids.size(dim=1) + out_len,
                                           do_sampling=self.do_sampling,
                                           top_k=self.top_k,
                                           temperature=self.temperature,
                                           num_return_sequences=self.num_return_sequences,
                                           num_beams=self.num_beams,
                                           early_stopping=self.early_stopping,)
        output = self.tokenizer.batch_decode(generate_ids,
                                             attention_mask=inputs.attention_mask,
                                             skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)[0]
        return output

    def debug_step(self, batch: Any):
        x, y = batch
        return y.output_tokens, x, y

    def step(self, batch: Any):
        prompt, example = batch
        out_len = self.tokenizer(example.output_tokens, return_tensors="pt").input_ids.size(dim=1)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        i = 0
        while inputs.input_ids.size(dim=1) + out_len >= 2048:
            i += 1
            prompt = down_sample_prompt(example, prompt)
            inputs = self.tokenizer(prompt, return_tensors="pt")
        if i > 0:
            print(f'WARNING: token length: {i} examples are removed from the prompt')
        output = self.forward(inputs, out_len)
        output_prompt = output.split(prompt)[-1].split('\n')[0]
        return output_prompt, prompt, example

    def training_step(self, batch: Any, batch_idx: int):
        prompt_choices, input_prompt, example = self.debug_step(batch)
        return {"output_prompt": prompt_choices, "input_prompt": input_prompt, "example": example}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_acc.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        prompt_choices, input_prompt, example = self.step(batch)
        return {"output_prompt": prompt_choices, "input_prompt": input_prompt, "example": example}

    def validation_epoch_end(self, outputs: List[Any]):
        return -1

    def test_step(self, batch: Any, batch_idx: int):
        prompt_choices, input_prompt, example = self.debug_step(batch)
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


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
