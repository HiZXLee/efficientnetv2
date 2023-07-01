import torch
from torch import nn
import yaml
import logging
from efficientnetv2_block import ConvBlock, FusedMBConvN, MBConvN

logger = logging.getLogger("effv2")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class EfficientNetV2(nn.Module):
    """A class for EfficientNetV2 model.

    Args:
        version (str, optional): EfficientNetV2 version. Supports ["s", "m", "l"] Defaults to "s".
        dropout_rate (float, optional): Dropout rate in the final classifier. Defaults to 0.2.
        num_classes (int, optional): Classes to be predicted. Defaults to 1000.

    """

    def __init__(self):
        super(EfficientNetV2, self).__init__()
        config = self.__config()
        self.version = config["version"]
        self.num_classes = config["num_classes"]
        self.dropout_rate = config["dropout_rate"]
        self.model_config_file = config["EFFICIENTNETV2_CONFIG"]

        last_channel = 1280
        self.features = self._feature_extractor(self.version, last_channel)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(self.dropout_rate, inplace=True),
            nn.Linear(last_channel, self.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

    def __config(self) -> dict:
        """To read configuration from config.yaml.

        Returns:
            dict: Dictionary of configuration.
        """
        with open("config.yaml", "r") as configFile:
            try:
                config = yaml.safe_load(configFile)
                return config
            except Exception as exc:
                logger.error(exc)

    def __model_config(self, model_config_file: str) -> dict:
        """To get EfficientNetV2 model configuration from model config yaml file.

        Args:
            model_config_file (str): model config yaml file name.

        Returns:
            dict: Configuration of EfficientNetV2 models.
        """
        with open(model_config_file, "r") as stream:
            try:
                Eff_V2_SETTINGS = yaml.safe_load(stream)
                return Eff_V2_SETTINGS
            except yaml.YAMLError as exc:
                logger.error(exc)

    def _feature_extractor(self, version, last_channel):
        Eff_V2_SETTINGS = self.__model_config(model_config_file=self.model_config_file)
      
        model_config = Eff_V2_SETTINGS[version]

        layers = []
        layers.append(ConvBlock(3, model_config[0][3], k_size=3, stride=2, padding=1))
        

        for (
            expansion_factor,
            k,
            stride,
            n_in,
            n_out,
            num_layers,
            use_fused,
        ) in model_config:
            if use_fused:
                layers += [
                    FusedMBConvN(
                        n_in if repeat == 0 else n_out,
                        n_out,
                        k_size=k,
                        stride=stride if repeat == 0 else 1,
                        expansion_factor=expansion_factor,
                    )
                    for repeat in range(num_layers)
                ]
            else:
                layers += [
                    MBConvN(
                        n_in if repeat == 0 else n_out,
                        n_out,
                        k_size=k,
                        stride=stride if repeat == 0 else 1,
                        expansion_factor=expansion_factor,
                    )
                    for repeat in range(num_layers)
                ]

        layers.append(ConvBlock(model_config[-1][4], last_channel, k_size=1))

        return nn.Sequential(*layers)


def __test():
    """A test function to show if the model is working properly. Will log the model output shape.
    """
   
    net = EfficientNetV2()
    x = torch.rand(4, 3, 224, 224)
    y = net(x)
    logger.info(f"Output shape: {y.shape}")


if __name__ == "__main__":
    __test()
