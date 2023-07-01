import pytest
import yaml
from yaml import YAMLError
from efficientnetv2 import EfficientNetV2
import torch


def test_yaml_file():
    """Unit test to check config.yaml and model config yaml is ok."""

    with open("config.yaml", "r") as configFile:
        try:
            config = yaml.safe_load(configFile)
            model_config_file = config["EFFICIENTNETV2_CONFIG"]
        except Exception as exc:
            pytest.fail(repr(exc))

    with open(model_config_file, "r") as stream:
        try:
            yaml.safe_load(stream)
            assert True
        except YAMLError as exc:
            pytest.fail(repr(exc))


def test_effv2(batch_size=4, num_classes=500):
    """Unit test to check if EfficientNetV2 model are returning desired output dimension.

    Args:
        batch_size (int, optional): Batch size of the input. Defaults to 4.
        num_classes (int, optional): Final total classes to predict. Defaults to 500.
    """
    
    net = EfficientNetV2()
    x = torch.rand(batch_size, 3, 224, 224)
    y = net(x)
    assert y.shape == torch.Size([batch_size, num_classes])
