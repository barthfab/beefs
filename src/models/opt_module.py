from transformers import AutoTokenizer, AutoModelForCausalLM, BloomTokenizerFast
from typing import Any, List
from datetime import datetime
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from src.utils.example_creators import parse_output_sentence_char, built_eval_doc, sort_nested_events, write_eval_file
from src.utils.event_evaluation import event_eval
from petals.client import DistributedBloomForCausalLM


class Opt(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        if 'bloom' in kwargs['model']:
            self.tokenizer = BloomTokenizerFast.from_pretrained(kwargs['model'], padding_side='right')
            self.model = DistributedBloomForCausalLM.from_pretrained(kwargs['model'])
        else:
            self.model = AutoModelForCausalLM.from_pretrained(kwargs['model'])
            self.tokenizer = AutoTokenizer.from_pretrained(kwargs['model'])
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
        if inputs.input_ids.size(dim=1) + 40 * 2 < 2024:
            generate_ids = self.model.generate(inputs.input_ids,
                                               attention_mask=inputs.attention_mask,
                                               max_length=inputs.input_ids.size(dim=1) + 40 * 2)
            output = self.tokenizer.batch_decode(generate_ids,
                                                 attention_mask=inputs.attention_mask,
                                                 skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)[0]
            prompt = self.tokenizer.batch_decode(inputs.input_ids,
                                                 attention_mask=inputs.attention_mask,
                                                 skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)[0]
            output_prompt = output.split(prompt)[-1].split('\n')[0]
        else:
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
        global_rec_error = 0
        nested_offset = 20
        output_dir = self.output + f"/{datetime.today().strftime('%Y-%m-%d-%H-%M')}"
        example_offsets = {}
        for output in tqdm(outputs):
            output_prompt = output['output_prompt']
            example = output['example']

            # extract all generated events
            pred, rec_error, rec_sentence = parse_output_sentence_char(example.tokens, output_prompt, example.nld)

            # sort output by single and nested events
            pred = sort_nested_events(pred)

            if example.id.split('_')[0] not in example_offsets.keys():
                example_offsets[example.id.split('_')[0]] = 0

            # build the evaluation files for the eval script
            file_lines, example_offset = built_eval_doc((pred, example),
                                                        event_types=[],
                                                        arg_finder=self.arg_finder,
                                                        example_offset=example_offsets[example.id.split('_')[0]])

            example_offsets[example.id.split('_')[0]] += len(pred) * nested_offset

            # write .a* file for evaluation
            write_eval_file(file_lines, output_dir, example.id.split('_')[0])

            # count reconstruction error
            if rec_error:
                global_rec_error += 1

        if outputs:
            # run corresponding evaluation script
            evaluator = event_eval(out_files=output_dir,
                                   builder_name=outputs[0]['example'].builder_name,
                                   split='devel')

            f1, prec, rec, unresolved_files = evaluator.eval()
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
