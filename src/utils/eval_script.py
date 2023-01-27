from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from src.utils.example_creators import parse_output_sentence_char, \
    built_eval_doc, sort_nested_events, write_eval_file, gold_list, parse_output_sentence
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


def local_eval(outputs, arg_finder):
    total_f1 = 0
    total_prec = 0
    total_rec = 0
    for output in outputs:
        gold = set()
        output_prompt = output['output_prompt']
        example = output['example']
        for event in example.events:
            gold.add(gold_list(example, event))

        if arg_finder == 0:
            # extract all generated events
            pred, rec_error, rec_sentence = parse_output_sentence_char(example.tokens, output_prompt, example.nld)
            result_set = events_only_results(pred, example)
        else:
            pred, rec_error = parse_output_sentence(output_prompt, example.nld)
            for special_token in [example.nld['begin_entity_token'], example.nld['end_entity_token']]:
                output_prompt = output_prompt.replace(special_token, ' ' + special_token + ' ')
            result_set = sentence_results(output_prompt)
        f1, prec, rec = calc_metric(result_set, gold)
        print(f'Output: {output["output_prompt"]} >>>> Gold: {output["example"].output_tokens} >>>> F1-Score: {f1}')
        total_rec += rec
        total_prec += prec
        total_f1 += f1
    total_f1 = total_f1/len(outputs)
    total_prec = total_prec / len(outputs)
    total_rec = total_rec / len(outputs)
    # sort output by single and nested events
    return total_f1, total_prec, total_rec


def events_only_results(pred, example):
    result_set = set()
    for prediction in pred:
        result = pred_list(pred, prediction, example)
        if result:
            result_set.add(result)
        else:
            miss_match_error = True
    return result_set


def sentence_results(output):
    print(1)
'''    for prediction in pred:
        head, tags, start, end = prediction
        if ' of ' in head:
            first = head.split(' of ')
            event_type = first[0]
            output_tuple = (event_type,)
            cut_out = " by ".join(" of ".join(first[1:]).split(' by ')[:-1])
            if ' Theme ' in cut_out:
                themes = head.split(' Theme ')
                for theme in themes:
                    if ' of ' in cut_out:
                        if ' and ' in theme:
                            output_theme = " and ".join(theme.split(' and '))
                        elif ' by ' in theme:
                            output_theme = theme.split(' by ')[0]
                    else:
                        if ' and ' in theme:
                            output_theme = theme.split(' and ')[0]
                            output_tuple += ('Theme', output_theme)
                        elif ' by ' in theme:
                            output_theme = theme.split(' by ')[0]
                            output_tuple += ('Theme', output_theme)
                        else:
                            output_tuple += ('Theme', theme)

            if ' Cause ' in cut_out:
        else:
            if ' by ' in head:
                return (head.split(' by ')[0],)
            return None


    return result_set

'''
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


def calc_metric(result_set, gold):
    result_intersection = gold.intersection(result_set)
    if len(gold) == len(result_set) == 0:
        return 1, 1, 1
    else:
        try:
            prec = len(result_intersection) / len(gold)
        except:
            if len(result_intersection) == 0:
                prec = 1
            else:
                prec = 0
        try:
            rec = len(result_intersection) / len(result_set)
        except:
            if len(gold) == 0:
                rec = 1
            else:
                rec = 0
        try:
            f1 = 2 * ((prec * rec) / (prec + rec))
        except:
            f1 = 0
        return f1, prec, rec