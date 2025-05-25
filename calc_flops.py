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

bert = getStandardModel().eval()
model = BertWrapper(bert)

# 创建 dummy 输入
input_ids = torch.ones((1, 128), dtype=torch.long)          # [batch_size, seq_length]
attention_mask = torch.ones((1, 128), dtype=torch.long)

# FLOPs 分析（注意传入是一个 tuple）
flops = FlopCountAnalysis(model, (input_ids, attention_mask))

# 输出
print(f"Estimated FLOPs: {flops.total() / 1e9:.2f} GFLOPs")