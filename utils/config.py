import os
from dataclasses import dataclass, asdict, astuple
from typing import Dict, Any
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

@dataclass
class Config:
    """
    Configuration settings for model training and the UAV environment.

    Attributes:
        device (str): Computation device ('cuda', 'mps', or 'cpu').
        hidden_N (int): Number of neurons in hidden layers.
        hidden_L (int): Number of hidden layers in the neural network.
        lr (float): Learning rate.
        random_seed (int): Random seed for reproducibility.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        num_users (int): Number of ground users.
        area_size (int): Size of the UAV environment area.
        height (int): Height of the UAV environment.
        beta_1 (float): Line-of-sight (LoS) channel parameter.
        beta_2 (float): Non-line-of-sight (NLoS) channel parameter.
        noise (float): Noise level.
        power (float): Transmission power.
        tanh_val (float): Hyper tangent value parameter.
        num_samples (int): Number of samples for the dataset.
        test_samples (int): Number of samples for testing.
        scaler (MinMaxScaler): Scaler for feature normalization.
        test_list (list[Any]): List of test values for hyperparameter experiments.
    """
    results_dir: str = ''
    
    device: str = 'cuda' if torch.cuda.is_available() else \
        'mps' if torch.backends.mps.is_available() else 'cpu'

    # Model settings
    hidden_N: int = 1024
    hidden_L: int = 4

    # Training settings
    lr: float = 1e-4
    random_seed: int = 42
    batch_size: int = 1024
    epochs: int = 10000

    # UAV and environment settings
    num_users: int = 4
    area_size: int = 200
    height: int = 70

    # Channel settings
    beta_1: float = 10 ** (-4.643)
    beta_2: float = 10 ** (-5.643)
    noise: float = 10 ** (-10.7)
    power: float = 1.0

    tanh_val: float = 0.2

    # Dataset settings
    num_samples: int = 500000
    test_samples: int = 10000

    # Scaler for feature normalization (initialized and fitted in __post_init__)
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Test settings for hyperparameter experiments
    test_list: list[Any] = None

    def __post_init__(self):
        """
        Post-initialization: Fit the scaler with dummy data based on area_size and num_users.

        The scaler is fitted with an array of shape (2, 2*num_users) where the values range
        from -area_size//2 to area_size//2.
        """
        self.scaler.fit(
            np.ones((2, 2 * self.num_users), dtype=np.float32) *
            np.array([[-self.area_size // 2, self.area_size // 2]]).T
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.

        :return: A dictionary representation of the configuration.
        """
        return asdict(self)

    def to_tuple(self) -> tuple[Any]:
        """
        Convert the configuration to a tuple.

        :return: A tuple representation of the configuration.
        """
        return astuple(self)

    def __str__(self) -> str:
        """
        Return the string representation of the configuration.

        :return: A string representation of the configuration.
        """
        return str(asdict(self))

    def replace(self, **kwargs) -> 'Config':
        """
        Create a new Config instance by replacing specified attributes.

        :param kwargs: Key-value pairs of attributes to replace.
        :return: A new Config instance with updated attributes.
        """
        base = self.to_dict()
        extra = {}
        for key, value in kwargs.items():
            if key in base:
                base[key] = value
            else:
                extra[key] = value
        new_instance = Config(**base)
        for key, value in extra.items():
            setattr(new_instance, key, value)
        return new_instance

    @classmethod
    def default(cls) -> 'Config':
        """
        Return the default configuration.

        :return: A Config instance with default values.
        """
        return cls()

    ################## Config generator for testing learning rates. ##################
    @classmethod
    def lr_test(cls) -> 'Config':
        """
        Create a configuration for testing learning rates.

        :return: A Config instance with test settings for learning rates.
        """
        return cls(results_dir=os.path.join('src', 'lr_test', 'result'),
                   num_samples=100000, epochs=1000,
                   test_list=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5])

    @classmethod
    def lr_test_gen(cls):
        """
        Generate configurations for testing different learning rates.

        :return: A generator that yields Config instances, each with a different learning rate from test_list.
        """
        base_cfg = cls.lr_test()
        for lr in base_cfg.test_list:
            yield base_cfg.replace(lr=lr)

    ############# Config generator for testing the number of ground users. #############
    @classmethod
    def gu_num_test(cls) -> 'Config':
        """
        Create a configuration for testing the number of ground users.

        :return: A Config instance with test settings for ground users.
        """
        return cls(results_dir=os.path.join('src', 'num_gu_test', 'result'),
                   epochs=1000, test_list=[2, 3, 4, 5, 6])

    @classmethod
    def gu_num_test_gen(cls):
        """
        Generate configurations for testing different numbers of ground users.

        :return: A generator that yields Config instances, each with a different number of ground users from test_list.
        """
        base_cfg = cls.gu_num_test()
        for gu in base_cfg.test_list:
            yield base_cfg.replace(num_users=gu)

    ############# Training models. #############
    @classmethod
    def training(cls) -> 'Config':
        """
        Generate a configuration for training model experiments.

        :return: A Config instance with training-specific settings.
        """
        return cls(results_dir=os.path.join('src', 'train_model', 'result'),
                   test_list=[[2, 3, 4, 5, 6], [50, 60, 70, 80, 90]])

    @classmethod
    def training_gen(cls, mode: str = 'num_gu'):
        """
        Generate training configurations with varying hyperparameters for model experiments.

        :return: A generator that yields Config instances with updated hyperparameter values.
        """
        base_cfg = cls.training()
        if mode == 'num_gu':
            for gu in base_cfg.test_list[0]:
                yield base_cfg.replace(num_users=gu, mode=mode)
        elif mode == 'height':
            for h in base_cfg.test_list[1]:
                yield base_cfg.replace(height=h, mode=mode)
        else:
            raise ValueError("Invalid mode. Choose 'num_gu' or 'height'.")

    @classmethod
    def brute_force(cls) -> 'Config':
        """
        Generate a configuration for brute-force model experiments.

        :return: A Config instance with brute-force-specific settings.
        """
        return cls(results_dir=os.path.join('src', 'brute_force', 'result'),
                   test_list=[[2, 3, 4, 5, 6], [50, 60, 70, 80, 90]]).replace()

    @classmethod
    def brute_force_gen(cls, mode: str = 'num_gu'):
        """
        Generate configurations for brute-force model experiments.

        :return: A generator that yields Config instances with updated hyperparameter values.
        """
        base_cfg = cls.brute_force()
        grid_step = 0.1
        chunk = 10_000
        if mode == 'num_gu':
            for gu in base_cfg.test_list[0]:
                yield base_cfg.replace(num_users=gu, mode=mode, grid_step=grid_step, chunk=chunk)
        elif mode == 'height':
            for h in base_cfg.test_list[1]:
                yield base_cfg.replace(height=h, mode=mode, grid_step=grid_step, chunk=chunk)
        else:
            raise ValueError("Invalid mode. Choose 'num_gu' or 'height'.")



def set_random_seed(cfg: Config = Config.default()) -> None:
    """
    Set random seeds for reproducibility in torch and numpy.

    :param cfg: A Config instance containing random_seed and device settings.
    """
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    if cfg.device == "cuda":
        torch.cuda.manual_seed_all(cfg.random_seed)
