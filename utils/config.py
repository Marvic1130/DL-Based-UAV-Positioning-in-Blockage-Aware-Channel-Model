from dataclasses import dataclass, asdict, astuple
from typing import Dict, Any
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

@dataclass
class Config:
    """
    Configuration settings for model training and UAV environment.

    Attributes:
        device (torch.device): Computation device ('cuda', 'mps', or 'cpu').
        hidden_N (int): Number of neurons in hidden layers.
        hidden_L (int): Number of hidden layers in the neural network.
        lr (float): Learning rate.
        random_seed (int): Random seed for reproducibility.
        batch (int): Batch size for training.
        epochs (int): Number of training epochs.
        num_users (int): Number of ground users.
        area_size (int): Size of the UAV environment area.
        beta_1 (float): LoS channel parameter.
        beta_2 (float): NLoS channel parameter.
        noise (float): Noise level.
        power (float): Transmission power.
        tanh_val (float): Hyper tangent value parameter.
        num_samples (int): Number of samples for the dataset.
        test_samples (int): Number of samples for testing.
    """
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else
                                        'mps' if torch.backends.mps.is_available() else 'cpu')

    # Model settings
    hidden_N: int = 1024  # Number of neurons in hidden layers
    hidden_L: int = 4  # Number of hidden layers in the neural network

    # Training settings
    lr: float = 1e-4  # Learning rate
    random_seed: int = 42
    batch: int = 1024  # Batch size for training
    epochs: int = 1000

    # UAV and environment settings
    num_users: int = 4  # Number of ground nodes
    area_size: int = 200
    height: int = 70

    # Channel settings
    beta_1: float = 10 ** (-4.643)  # LoS
    beta_2: float = 10 ** (-5.643)  # NLoS
    noise: float = 10 ** (-10.7)
    power: float = 1.0

    tanh_val: float = 0.2

    # Dataset settings
    num_samples: int = 500000
    test_samples: int = 10000

    scaler = MinMaxScaler(feature_range=(0, 1))

    # Test settings
    test_list: list[Any] = None

    def __post_init__(self):
        self.scaler.fit(
            np.ones((2, 2 * self.num_users), dtype=np.float32) *
            np.array([[-self.area_size // 2, self.area_size // 2]]).T
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        :return: A dictionary representation of the configuration.
        """
        return asdict(self)

    def to_tuple(self) -> tuple[Any]:
        """
        :return: A tuple representation of the configuration.
        """
        return astuple(self)

    def __str__(self) -> str:
        """
        :return: A string representation of the configuration.
        """
        return str(asdict(self))

    def replace(self, **kwargs) -> 'Config':
        """
        Replace the attributes of the configuration.

        :param kwargs: Attributes to replace.
        :return: A new Config instance with the replaced attributes.
        """
        return Config(**{**self.to_dict(), **kwargs})

    ################## Configure the default settings. ##################

    @classmethod
    def default(cls):
        """
        Return the default configuration.

        :return: A Config instance with default values.
        """
        return cls()

    ################## Config generator for testing learning rates. ##################
    @classmethod
    def lr_test(cls):
        """
        Config for testing learning rates.

        :return: A Config instance for testing learning rates.
        """
        return cls(num_samples=100000, epochs=1000, test_list=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5])

    @classmethod
    def lr_test_gen(cls):
        """
        Config generator for testing learning rates.

        :return: A generator that yields Config instances, each with a different learning rate from lr_ls.
        """
        base_cfg = cls.lr_test()
        for lr in base_cfg.test_list:
            yield base_cfg.replace(lr=lr)

    ############# Config generator for testing the number of ground nodes. #############

    @classmethod
    def gu_num_test(cls):
        """
        Config for testing the number of ground nodes.

        :return: A Config instance for testing the number of ground nodes.
        """
        return cls(num_samples=100000, epochs=1000, test_list=[2, 3, 4, 5, 6])

    @classmethod
    def gu_num_test_gen(cls):
        """
        Config generator for testing the number of ground nodes.

        :return: A generator that yields Config instances, each with a different number of ground users from gu_num_ls.
        """
        base_cfg = cls.gu_num_test()
        for gu_num in base_cfg.test_list:
            yield cls(num_users=gu_num)


def set_random_seed(cfg: Config = Config.default()):
    """
    Set random seeds for reproducibility.

    :param cfg: Configuration settings.
    """
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    if cfg.device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.random_seed)
