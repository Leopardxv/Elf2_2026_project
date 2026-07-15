import pandas
import rknn
import seaborn
import torch
import torchvision
from pathlib import Path

assert torch.cuda.is_available()
print("training-environment-ok")
print(torch.__version__)
print(torchvision.__version__)
Path(__file__).with_name(".tooling").joinpath("training-env-ok.txt").write_text(
    f"torch={torch.__version__}\ntorchvision={torchvision.__version__}\n",
    encoding="utf-8",
)
