from transformers import AutoTokenizer, AutoModelForCausalLM, BloomTokenizerFast
from typing import Any, List
import pytorch_lightning as pl
from src.utils.eval_script import a2_evaluation, local_eval


class Opt(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(kwargs['model'])
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs['model'])
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

    def forward(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.device)
        if inputs.input_ids.size(dim=1) + 40 <= 2048:
            generate_ids = self.model.generate(inputs.input_ids,
                                               attention_mask=inputs.attention_mask,
                                               max_length=inputs.input_ids.size(dim=1) + 48,
                                               top_k=self.top_k,
                                               temperature=self.temperature,
                                               top_p=self.top_p,)
            output = self.tokenizer.batch_decode(generate_ids,
                                                 attention_mask=inputs.attention_mask,
                                                 skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)[0]
            output_prompt = output.split(prompt)[-1].split('\n')[0]
        else:
            print(f'WARNING: token length: {inputs.input_ids.size(dim=1)}')
            output_prompt = ""
        return output_prompt

    def step(self, batch: Any):
        x, y = batch
        output_prompt = self.forward(x)
        return output_prompt, x, y

    def training_step(self, batch: Any, batch_idx: int):
        prompt_choices, input_prompt, example = self.step(batch)
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
        prompt_choices, input_prompt, example = self.step(batch)
        return {"output_prompt": prompt_choices, "input_prompt": input_prompt, "example": example}

    def test_epoch_end(self, outputs: List[Any]):
        if self.local_eval:
            f1, prec, rec = local_eval(outputs, self.arg_finder)
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
