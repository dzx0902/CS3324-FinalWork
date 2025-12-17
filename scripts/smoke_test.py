import torch
from src.models.iqa_heads.resnet_baseline import ResNetBaselineIQA
from src.models.iqa_heads.stair_iqa import StairIQA
from src.models.iqa_heads.hyper_iqa import HyperIQA
from src.models.iqa_heads.caqf_iqa import CAQF_IQA, CAQF_IQA_NoAttn
from src.models.iqa_heads.tiny_iqa import TinyIQA_R18


def run():
    x = torch.randn(1, 3, 224, 224)
    for cls in [ResNetBaselineIQA, StairIQA, HyperIQA, CAQF_IQA, CAQF_IQA_NoAttn, TinyIQA_R18]:
        m = cls()
        y = m(x)
        print(cls.__name__, y.shape, y.detach().numpy())


if __name__ == "__main__":
    run()

