import yaml
import os
import os.path
import glob
from easydict import EasyDict
import pandas as pd

from utils.utils import recreate_dirs

from typing import Union

REPO_ROOT = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", ".."))


def search_for_config_yaml_file(cfg_id: str) -> str:
    search_path = os.path.join(REPO_ROOT, 'cfg/**/%s.yml' % cfg_id)
    found_files = glob.glob(search_path, recursive=True)
    assert len(found_files) == 1, "Couldn't find the configuration file."
    return os.path.abspath(found_files[0])


class Config:

    def __init__(self, cfg_id: Union[str, os.PathLike]):
        if os.path.isfile(cfg_id) and os.path.exists(cfg_id):
            cfg_path = os.path.abspath(cfg_id)
        else:
            cfg_path = search_for_config_yaml_file(cfg_id)
        assert os.path.exists(cfg_path)

        self.id = os.path.splitext(os.path.basename(cfg_path))[0]
        self.yml_dict = EasyDict(yaml.safe_load(open(cfg_path, 'r')))

    def __getattribute__(self, name):
        yml_dict = super().__getattribute__('yml_dict')
        if name in yml_dict:
            return yml_dict[name]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        try:
            yml_dict = super().__getattribute__('yml_dict')
        except AttributeError:
            return super().__setattr__(name, value)
        if name in yml_dict:
            yml_dict[name] = value
        else:
            return super().__setattr__(name, value)

    def get(self, name, default=None):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            return default


class ModelConfig(Config):

    def __init__(self, cfg_id: Union[str, os.PathLike], tmp: bool = False, create_dirs: bool = False):
        Config.__init__(self, cfg_id=cfg_id)

        # data dir
        self.results_root_dir = os.path.join(REPO_ROOT, self.yml_dict['results_root_dir'])
        # results dirs
        cfg_root_dir = '/tmp/agentformer' if tmp else self.results_root_dir
        self.cfg_root_dir = os.path.expanduser(cfg_root_dir)

        self.cfg_dir = os.path.join(self.cfg_root_dir, self.id)
        self.model_dir = os.path.join(self.cfg_dir, 'models')
        self.result_dir = os.path.join(self.cfg_dir, 'results')
        self.log_dir = os.path.join(self.cfg_dir, 'log')
        self.tb_dir = os.path.join(self.cfg_dir, 'tb')
        self.model_path = os.path.join(self.model_dir, 'model_%s.p')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        if create_dirs:
            recreate_dirs(self.tb_dir)

    def get_best_val_checkpoint_name(self):
        if self.yml_dict['model_id'] in {'const_velocity', 'oracle'}:
            return 'untrained'

        val_csv_path = os.path.join(self.model_dir, 'models.csv')
        assert os.path.exists(val_csv_path)
        val_csv = pd.read_csv(val_csv_path)
        val_csv = val_csv[~(val_csv == val_csv.columns).all(axis=1)]
        val_csv['val_loss'] = val_csv['val_loss'].astype(float)

        checkpoint_name = str(val_csv[val_csv['val_loss'] == val_csv['val_loss'].min()]['model_name'].item())
        return checkpoint_name
