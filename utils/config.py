import os
from typing import Final
import yaml
import torch

class _Hyperparameters:
    def __init__(self):
        try:
            with open('utils/config.yaml', 'r') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            raise Exception("Configuration file not found.")
        except yaml.YAMLError:
            raise Exception("Error parsing YAML file.")

        # Model settings
        if config['device'] == 'auto':
            config['device'] = torch.device('cuda' if torch.cuda.is_available() else
                                            'mps' if torch.backends.mps.is_available() else 'cpu')
        super().__setattr__('_device', config['device'])
        super().__setattr__('_batch', self._safe_eval(config['batch']))
        super().__setattr__('_epochs', self._safe_eval(config['epochs']))
        super().__setattr__('_num_node', self._safe_eval(config['num_node']))
        super().__setattr__('_num_time', self._safe_eval(config['num_time']))
        super().__setattr__('_max_dist', self._safe_eval(config['max_dist']))
        super().__setattr__('_area_size', self._safe_eval(config['area_size']))
        super().__setattr__('_v_speed', self._safe_eval(config['v_speed']))
        super().__setattr__('_H_min', self._safe_eval(config['H_min']))
        super().__setattr__('_H_max', self._safe_eval(config['H_max']))
        super().__setattr__('_num_blocks', self._safe_eval(config['num_blocks']))
        super().__setattr__('_beta_1', self._safe_eval(config['beta_1']))
        super().__setattr__('_beta_2', self._safe_eval(config['beta_2']))
        super().__setattr__('_noise', self._safe_eval(config['noise']))
        super().__setattr__('_P_AVG', self._safe_eval(config['P_AVG']))
        super().__setattr__('_P_PEAK', self._safe_eval(config['P_PEAK']))
        super().__setattr__('_num_samples', self._safe_eval(config['num_samples']))
        super().__setattr__('_num_val_samples', self._safe_eval(config['num_val_samples']))
        super().__setattr__('_hidden_N', self._safe_eval(config['hidden_N']))
        super().__setattr__('_hidden_L', self._safe_eval(config['hidden_L']))
        super().__setattr__('_penalty_weight', [self._safe_eval(penalty) for penalty in config['penalty_weight']])
        super().__setattr__('_checkpoint_path', config['checkpoint_path'])
        super().__setattr__('_output_directory', config['output_directory'])

    def _safe_eval(self, value):
        """Safely evaluate a value if it's a string."""
        if isinstance(value, str):
            try:
                # Only eval if necessary, and catch errors in case of invalid expressions
                return eval(value)
            except (NameError, SyntaxError):
                raise ValueError(f"Invalid expression in configuration: {value}")
        return value

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise TypeError(f'Cannot reassign immutable attribute: {name}')
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self.__dict__:
            raise Exception(f'Cannot delete immutable attribute: {name}')
        super().__delattr__(name)

    @property
    def device(self):
        return self._device

    @property
    def batch(self):
        return self._batch

    @property
    def epochs(self):
        return self._epochs

    @property
    def num_node(self):
        return self._num_node

    @property
    def num_time(self):
        return self._num_time

    @property
    def max_dist(self):
        return self._max_dist

    @property
    def area_size(self):
        return self._area_size

    @property
    def v_speed(self):
        return self._v_speed

    @property
    def H_min(self):
        return self._H_min

    @property
    def H_max(self):
        return self._H_max

    @property
    def num_blocks(self):
        return self._num_blocks

    @property
    def beta_1(self):
        return self._beta_1

    @property
    def beta_2(self):
        return self._beta_2

    @property
    def noise(self):
        return self._noise

    @property
    def P_AVG(self):
        return self._P_AVG

    @property
    def P_PEAK(self):
        return self._P_PEAK

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def num_val_samples(self):
        return self._num_val_samples

    @property
    def hidden_N(self):
        return self._hidden_N

    @property
    def hidden_L(self):
        return self._hidden_L

    @property
    def penalty_weight(self):
        return self._penalty_weight

    @property
    def checkpoint_path(self):
        return self._checkpoint_path

    @property
    def output_directory(self):
        return self._output_directory

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


Hyperparameters = _Hyperparameters()

if __name__ == "__main__":


    # Example Usage
    print(Hyperparameters.device)
    print(Hyperparameters.penalty_weight)