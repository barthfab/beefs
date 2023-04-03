from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from src.utils.example_creators import parse_output_sentence_char, \
    built_eval_doc, sort_nested_events, write_eval_file, gold_list, nested_output_converter
from src.utils.a2_evaluation_class import event_eval

PATH = Path(__file__).parent.parent.parent.resolve()

#todo adapt for gpt3
def a2_evaluation(outputs, output, arg_finder):
    global_rec_error = 0
    nested_offset = 20
    output_dir = output + f"/{datetime.today().strftime('%Y-%m-%d-%H-%M')}"
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
                                                    arg_finder=arg_finder,
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
        return f1, prec, rec
    return -1


def local_evaluation(outputs, arg_finder):
    micro_gold = []
    micro_pred = []
    for output in outputs:
        gold = set()
        output_prompt = output['output_prompt']
        example = output['example']
        for event in example.events:
            gold.add(gold_list(example, event))
        if arg_finder < 2:
            # extract all generated events
            pred, rec_error, rec_sentence = parse_output_sentence_char(example.tokens, output_prompt, example.nld)
            if arg_finder == 1:
                #result converter for only events without nested nld
                result_set = events_only_results(pred, example,)
            else:
                #result converter for plain text modified nld
                result_set = events_only_results(pred, example, plain=True)
        else:
            #result converter for nested nld with arg_types first
            conv = nested_output_converter()
            result_set = conv.convert_arg_start(output_prompt, example.nld)
        micro_gold.append(gold)
        micro_pred.append(result_set)
        print(f'Output: {output["output_prompt"]} >>>> Gold: {output["example"].output_tokens}')
        print(f"Pred_Sets: {result_set} >>>> Gold_sets: {gold}")
    f1, prec, rec = calc_metric(micro_pred, micro_gold)
    # sort output by single and nested events
    return f1, prec, rec


def events_only_results(pred, example, plain: bool = False):
    result_set = set()
    for prediction in pred:
        if not plain:
            result = pred_list(pred, prediction, example)
        else:
            result = plain_pred_list(pred, prediction, example)
        if result:
            result_set.add(result)
        else:
            miss_match_error = True
    return result_set

def plain_pred_list(preds, pred, example):
    head, tags, start, end = pred
    event_tuple = (tags[0][0],)
    preds_names = [e[0] for e in preds]
    if len(tags) == 1:
        return event_tuple
    for arg in tags[1:]:
        try:
            arg_name, arg_type = arg
        except:
            continue
        if arg_name in [example.tokens[pred_names.start:pred_names.end] for pred_names in example.entities]:
            new_event = [e for e in example.entities if example.tokens[e.start:e.end] == arg_name]
            if new_event:
                event_tuple += (arg_type, example.tokens[new_event[0].start:new_event[0].end])
        if arg_name in preds_names:
            new_event = [e for e in preds if e[0] == arg_name and e != pred]
            if new_event:
                if new_event[0][0] == head:
                    # get a random event that has not the same trigger as the current
                    x = [e for e in new_event if e[-1] != end and e[-2] != end]
                    if not x:
                        x = [e for e in new_event if e[1][0][0] != tags[0][0]]
                        if not x:
                            x = new_event
                    if [e for e in x if head not in [a[0] for a in e[1][1:]]]:
                        x = [e for e in x if head not in [a[0] for a in e[1][1:]]]
                    less_preds = [p for p in preds if head != p[0]]
                    less_preds.append(x[0])
                    event_tuple = event_tuple + (arg_type, plain_pred_list(less_preds, new_event[0], example))
                else:
                    if [e for e in new_event if head not in [a[0] for a in e[1][1:]]]:
                        new_event = [e for e in new_event if head not in [a[0] for a in e[1][1:]]]
                    event_tuple = event_tuple + (arg_type, plain_pred_list(preds, new_event[0], example))
    return event_tuple


def pred_list(preds, pred, example):
    head, tags, start, end = pred
    pred_id, pred_type = head.split(':')
    event_tuple = (pred_type,)
    if not tags:
        return (event_tuple,)
    for arg in tags:
        try:
            arg_id, arg_type = arg
        except:
            continue
        if arg_id.startswith('Ev'):
            if not arg_id == pred_id:
                new_event = [e for e in preds if e[0].split(':')[0] == arg_id]
                if new_event:
                    event_tuple = event_tuple + (arg_type, pred_list(preds, new_event[0], example))
        else:
            new_event = [e for e in example.entities if example.tokens[e.start:e.end] == arg_id]
            if new_event:
                event_tuple += (arg_type, example.tokens[new_event[0].start:new_event[0].end])
    return event_tuple


def calc_metric(micro_pred, micro_gold):
    intersec = 0
    gold_result = 0
    pred_result = 0
    for pred, gold in zip(micro_pred, micro_gold):
        result_intersection = gold.intersection(pred)
        intersec += len(result_intersection)
        gold_result += len(gold)
        pred_result += len(pred)
    try:
        prec = intersec / gold_result
    except:
        if intersec == 0:
            prec = 1
        else:
            prec = 0
    try:
        rec = intersec / pred_result
    except:
        if gold_result == 0:
            rec = 1
        else:
            rec = 0
    try:
        f1 = 2 * ((prec * rec) / (prec + rec))
    except:
        f1 = 0
    return f1, prec, rec
