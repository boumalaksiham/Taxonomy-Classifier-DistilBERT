"""Execute the trainer's checkpoint-selection statements with controlled epochs.

No torch imports, model downloads, or claims about learned model quality.
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
KIND = 'taxonomy'

class CheckpointTests(unittest.TestCase):
    def run_selection(self, scores):
        tree = ast.parse((ROOT / 'train.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        start = next(i for i,n in enumerate(main.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id.startswith('best_') for t in n.targets))
        end = next(i for i,n in enumerate(main.body[start:], start) if isinstance(n, ast.Expr)
                   and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                   and n.value.func.attr == 'load_state_dict')
        statements = main.body[start:end+1]
        model = SimpleNamespace(epoch=0)
        model.state_dict = lambda: {'epoch':model.epoch}
        def restore(state): model.epoch = state['epoch']
        model.load_state_dict = restore
        stored = {}
        saves = []
        def save(state, path):
            stored[path] = copy.deepcopy(state); saves.append(state['epoch'])
        def train(*args): model.epoch += 1; return 0.1
        def evaluate(*args):
            score = scores[model.epoch-1]
            return (score, score, score) if KIND == 'taxonomy' else (score, [], [])
        namespace = {'CONFIG':{'epochs':len(scores)},'model':model,'train_loader':None,
            'validation_loader':None,'optimizer':None,'criterion':None,'device':'cpu',
            'id_to_label':{},'train_epoch':train,'evaluate':evaluate,'os':Mock(),
            'torch':SimpleNamespace(save=save,load=lambda path,**kwargs:copy.deepcopy(stored[path])),
            'print':lambda *args,**kwargs:None}
        exec(compile(ast.fix_missing_locations(ast.Module(body=statements,type_ignores=[])),
                     str(ROOT/'train.py'),'exec'),namespace)
        return model.epoch,saves

    def test_reports_can_use_selected_checkpoint_instead_of_last_epoch(self):
        scores = [.6,.2,.5] if KIND == 'image' else [0,.8,.2]
        selected,saves = self.run_selection(scores)
        self.assertEqual(selected,2)
        self.assertEqual(saves,[1,2])

    def test_first_finite_checkpoint_is_saved_when_scores_tie(self):
        selected,saves = self.run_selection([0,0,0])
        self.assertEqual(selected,1)
        self.assertEqual(saves,[1])

if __name__ == '__main__': unittest.main()
