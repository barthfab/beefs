import os
import pandas
import subprocess
import torch.distributed as dist
from pathlib import Path

PATH = Path(__file__).parent.parent.parent.resolve()


class event_eval:
    def __init__(self, out_files: str, builder_name: str, split: str, source_files: str = "eval_source/", ):

        # all files that should be evaluated
        self.files = os.listdir(Path.joinpath(PATH, out_files, 'orig'))

        # path of the files
        self.orig_path = Path.joinpath(PATH, out_files, 'orig')

        # path for the evaluated files
        self.evaluated_path = Path.joinpath(PATH, out_files, 'evaluated')

        # path of the gold files
        self.source = Path.joinpath(PATH.parent, source_files, builder_name)

        # dataset name
        self.builder_name = builder_name
        self.split = split

        self.advanced_output = {}

    def bionlp_st_2013_pc(self):
        unresolved_files = []
        for file in self.files:
            if file not in os.listdir(f'{self.source}/{self.split}'):
                continue
            # test for single file if eval script works
            os.system(
                f'python2 {PATH}/src/utils/pc_eval.py -r {self.source}/{self.split} -o {self.evaluated_path} {self.orig_path}/{file}')
            eval_files = os.listdir(f'{self.evaluated_path}')
            # if file was not evaluated remove it
            if file not in eval_files:
                os.rename(f"{self.orig_path}/{file}",
                          f"{Path.joinpath(self.orig_path.parent, str(file))}")
                unresolved_files.append(file)

        # eval all remaining files
        os.system(f'python2 {PATH}/src/utils/pc_eval.py -r {self.source}/{self.split} -o {self.evaluated_path} {self.orig_path}/*')

        # find stats in csv
        if os.path.isfile(f'{self.evaluated_path}/stats.csv'):
            results = pandas.read_csv(f'{self.evaluated_path}/stats.csv', sep='\t')
            f1 = results["F-score"].mean()
            prec = results["precision"].mean()
            rec = results["recall"].mean()
        else:
            f1, prec, rec = 0, 0, 0
        '''
        # put all removed files back in place
        for file in unresolved_files:
            os.rename(f"{Path.joinpath(self.orig_path.parent, str(file))}",
                      f"{self.orig_path}/{file}")
        '''
        return f1, prec, rec, unresolved_files

    def bionlp_st_2013_ge(self):
        # not implemented yet
        return -1

    def bionlp_st_2011_ge(self):
        unresolved_files = []

        # check for every file if the script works
        for file in self.files:
            if file not in os.listdir(f'{self.source}/{self.split}'):
                continue
            p = subprocess.Popen([f'perl', f'{PATH}/src/utils/ge_eval.pl', '-g', f'{self.source}/{self.split}',
                                  f'{self.orig_path}/{file}'], shell=False)
            # kill files that are not working
            try:
                p.wait(2)
            except subprocess.TimeoutExpired:
                p.kill()
                # remove corrupted files
                os.rename(f"{self.orig_path}/{file}",
                          f"{Path.joinpath(self.orig_path.parent, str(file))}")
                unresolved_files.append(file)

        # run eval script on all remaining files
        output_promt = os.popen(
            f'perl {PATH}/src/utils/ge_eval.pl -g {self.source}/{self.split} {self.orig_path}/*').readlines()
        f1, prec, rec = 0, 0, 0

        # read output to log the score
        for line in output_promt[3:]:
            if line.startswith('-') or 'Event Class' in line:
                continue
            fields = list(filter(None, line.split(' ')))
            name = fields[0]
            if 'TOTAL' in name:
                if 'ALL-TOTAL' in name:
                    f1 = float(fields[-1][:-2])
                    prec = float(fields[-2])
                    rec = float(fields[-3])
                else:
                    spec = name.split('=[')
                    self.advanced_output[f'{spec[0][0:2]}-total'] = (float(fields[-1][:-2]), float(fields[-2]), float(fields[-3]))
            else:
                self.advanced_output[name] = (float(fields[-1][:-2]), float(fields[-2]), float(fields[-3]))
        '''
        # put all removed files back in place
        for file in unresolved_files:
            os.rename(f"{Path.joinpath(self.orig_path.parent, str(file))}",
                      f"./{self.orig_path}/{file}")
        '''
        return f1, prec, rec, unresolved_files

    def eval(self):
        return self.EVAL_SCRIPTS[self.builder_name](self)

    def return_advanced_output(self):
        return self.advanced_output

    EVAL_SCRIPTS = {'bionlp_st_2013_pc': bionlp_st_2013_pc,
                    'bionlp_st_2013_ge': bionlp_st_2013_ge,
                    'bionlp_st_2011_ge': bionlp_st_2011_ge,
                    }
