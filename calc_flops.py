import logging
from transformers import BertModel
from fvcore.nn import FlopCountAnalysis
from hardPrune.qnli.standardise import getStandardModel
import torch

class BertWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


def calc_flops(_model):
    bert = _model.eval().cuda()
    model = BertWrapper(bert).cuda()
    input_ids = torch.ones((1, 128), dtype=torch.long).cuda()
    attention_mask = torch.ones((1, 128), dtype=torch.long).cuda()

    flops = FlopCountAnalysis(model, (input_ids, attention_mask))

    logging.getLogger(__name__).info(f"Estimated FLOPs: {flops.total() / 1e9:.2f} GFLOPs")