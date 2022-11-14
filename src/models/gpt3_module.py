import openai
from typing import Any, List
from datetime import datetime
from tqdm import tqdm
import pytorch_lightning as pl
from src.utils.example_creators import parse_output_sentence_char, built_eval_doc, sort_nested_events, \
    built_eval_doc_sorted, write_eval_file
from src.utils.event_evaluation import event_eval


class Gpt3(pl.LightningModule):
    def __init__(self, model, max_tokens, temperature, top_p, n, stream,
                 logprobs, presence_penalty, frequency_penalty, stop, api_key, output: str = 'logs/train_result', arg_finder=0):
        super().__init__()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stream = stream
        self.logprobs = logprobs
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.stop = stop
        self.output = output
        self.arg_finder = arg_finder
        openai.api_key = api_key

    def forward(self, x: str):
        return openai.Completion.create(model=self.model,
                                        prompt=x,
                                        max_tokens=self.max_tokens,
                                        temperature=self.temperature,
                                        top_p=self.top_p,
                                        n=self.n,
                                        stream=self.stream,
                                        logprobs=self.logprobs,
                                        presence_penalty=self.presence_penalty,
                                        frequency_penalty=self.frequency_penalty,
                                        stop=self.stop, )

    def debug_step(self, batch: any):
        x, y = batch
        return y.output_tokens, x, y

    def step(self, batch: Any):
        x, y = batch
        output_prompt = self.forward(x)
        return output_prompt.choices.text, x, y

    def training_step(self, batch: Any, batch_idx: int):
        #todo
        prompt_choices, input_prompt, example = self.debug_step(batch)
        return {"output_prompt": prompt_choices, "input_prompt": input_prompt, "example": example}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_acc.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        prompt_choices, input_prompt, example = self.debug_step(batch)
        return {"output_prompt": prompt_choices, "input_prompt": input_prompt, "example": example}

    def validation_epoch_end(self, outputs: List[Any]):
        return -1

    def test_step(self, batch: Any, batch_idx: int):
        prompt_choices, input_prompt, example = self.debug_step(batch)
        return {"output_prompt": prompt_choices, "input_prompt": input_prompt, "example": example}

    def test_epoch_end(self, outputs: List[Any]):
        # todo safe all found events in a dict
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
            file_lines, example_offset = built_eval_doc_sorted((pred, example),
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
            #todo safe all found events in a dict

    def configure_optimizers(self):
        return None


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
