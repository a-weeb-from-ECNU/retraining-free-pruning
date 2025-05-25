import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification, BertModel

class CustomBertLayer(nn.Module):
    def __init__(self, config, layer_id, keep_heads, keep_neurons):
        super().__init__()
        self.keep_heads = keep_heads
        self.keep_neurons = keep_neurons

        self.num_heads = len(keep_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.attention_dim = self.num_heads * self.head_dim

        # Attention
        self.query = nn.Linear(config.hidden_size, self.attention_dim)
        self.key = nn.Linear(config.hidden_size, self.attention_dim)
        self.value = nn.Linear(config.hidden_size, self.attention_dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.att_out = nn.Linear(self.attention_dim, config.hidden_size)
        self.att_norm = nn.LayerNorm(config.hidden_size)

        # FFN
        self.intermediate = nn.Linear(config.hidden_size, len(keep_neurons))
        self.output = nn.Linear(len(keep_neurons), config.hidden_size)
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.activation = nn.GELU()

    def forward(self, x):
        # Self-attention (simplified, no masking)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        B, T, _ = Q.shape
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # 修复：添加括号
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = attn_probs @ V
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        attn_output = self.dropout(self.att_out(attn_output))
        x = self.att_norm(x + attn_output)

        # FFN
        h = self.activation(self.intermediate(x))
        h = self.output(h)
        x = self.output_norm(x + h)
        return x


class PrunedBertForSequenceClassification(nn.Module):
    def __init__(self, config, head_mask, neuron_mask):
        super().__init__()
        self.config = config
        self.head_mask = head_mask
        self.neuron_mask = neuron_mask

        # 修复1：正确创建embeddings
        bert_model = BertModel(config)
        self.embeddings = bert_model.embeddings

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            keep_heads = head_mask[i].nonzero(as_tuple=True)[0].tolist()
            keep_neurons = neuron_mask[i].nonzero(as_tuple=True)[0].tolist()
            layer = CustomBertLayer(config, i, keep_heads, keep_neurons)
            self.layers.append(layer)

        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_act = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # 修复2：正确处理embeddings的输出
        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        
        # 修复3：检查embedding_output的类型
        if hasattr(embedding_output, 'last_hidden_state'):
            x = embedding_output.last_hidden_state
        else:
            x = embedding_output

        for layer in self.layers:
            x = layer(x)

        pooled = self.pooler_act(self.pooler(x[:, 0]))  # Use [CLS] token
        logits = self.classifier(pooled)
        return logits


def load_and_transfer_weights(model, state_dict, head_mask, neuron_mask):
    """
    将原始BERT权重转移到剪枝后的模型中
    """
    print("开始权重转移...")
    
    # 获取配置信息
    config = model.config
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads
    
    print(f"模型配置: hidden_size={config.hidden_size}, num_heads={num_heads}, head_dim={head_dim}")
    
    # 转移embedding权重
    embedding_keys = [k for k in state_dict.keys() if k.startswith('bert.embeddings')]
    for key in embedding_keys:
        new_key = key.replace('bert.embeddings.', 'embeddings.')
        if new_key in model.state_dict():
            model.state_dict()[new_key].copy_(state_dict[key])
            print(f"转移权重: {key} -> {new_key}")
    
    # 转移每层的权重
    for layer_idx in range(len(model.layers)):
        keep_heads = head_mask[layer_idx].nonzero(as_tuple=True)[0].tolist()
        keep_neurons = neuron_mask[layer_idx].nonzero(as_tuple=True)[0].tolist()
        
        print(f"Layer {layer_idx}: 保留 {len(keep_heads)}/{num_heads} 个注意力头, {len(keep_neurons)} 个FFN神经元")
        
        # 注意力权重转移 (QKV)
        for weight_type in ['query', 'key', 'value']:
            old_key = f'bert.encoder.layer.{layer_idx}.attention.self.{weight_type}.weight'
            if old_key in state_dict:
                old_weight = state_dict[old_key]  # shape: [768, 768]
                print(f"原始{weight_type}权重形状: {old_weight.shape}")
                
                # 重塑为 (num_heads, head_dim, hidden_size) 
                old_weight_reshaped = old_weight.view(num_heads, head_dim, config.hidden_size)
                # 选择保留的头
                new_weight = old_weight_reshaped[keep_heads]  # shape: [len(keep_heads), head_dim, hidden_size]
                # 重塑回 (new_attention_dim, hidden_size)
                new_weight = new_weight.view(-1, config.hidden_size)
                
                print(f"新{weight_type}权重形状: {new_weight.shape}")
                print(f"目标层权重形状: {model.layers[layer_idx].__getattr__(weight_type).weight.shape}")
                
                if new_weight.shape == model.layers[layer_idx].__getattr__(weight_type).weight.shape:
                    model.layers[layer_idx].__getattr__(weight_type).weight.data.copy_(new_weight)
                else:
                    print(f"警告: {weight_type}权重形状不匹配，跳过")
        
        # 注意力偏置转移 (QKV)
        for weight_type in ['query', 'key', 'value']:
            old_key = f'bert.encoder.layer.{layer_idx}.attention.self.{weight_type}.bias'
            if old_key in state_dict:
                old_bias = state_dict[old_key]  # shape: [768]
                # 重塑为 (num_heads, head_dim)
                old_bias_reshaped = old_bias.view(num_heads, head_dim)
                # 选择保留的头
                new_bias = old_bias_reshaped[keep_heads].view(-1)
                
                if new_bias.shape == model.layers[layer_idx].__getattr__(weight_type).bias.shape:
                    model.layers[layer_idx].__getattr__(weight_type).bias.data.copy_(new_bias)
        
        # 注意力输出权重转移
        old_key = f'bert.encoder.layer.{layer_idx}.attention.output.dense.weight'
        if old_key in state_dict:
            old_weight = state_dict[old_key]  # shape: [768, 768]
            # 重塑为 (hidden_size, num_heads, head_dim)
            old_weight_reshaped = old_weight.view(config.hidden_size, num_heads, head_dim)
            # 选择保留的头
            new_weight = old_weight_reshaped[:, keep_heads, :].contiguous().view(config.hidden_size, -1)
            
            if new_weight.shape == model.layers[layer_idx].att_out.weight.shape:
                model.layers[layer_idx].att_out.weight.data.copy_(new_weight)
        
        # 注意力输出偏置转移
        old_key = f'bert.encoder.layer.{layer_idx}.attention.output.dense.bias'
        if old_key in state_dict:
            old_bias = state_dict[old_key]
            if old_bias.shape == model.layers[layer_idx].att_out.bias.shape:
                model.layers[layer_idx].att_out.bias.data.copy_(old_bias)
        
        # 注意力LayerNorm转移
        for norm_param in ['weight', 'bias']:
            old_key = f'bert.encoder.layer.{layer_idx}.attention.output.LayerNorm.{norm_param}'
            if old_key in state_dict:
                old_param = state_dict[old_key]
                if old_param.shape == model.layers[layer_idx].att_norm.__getattr__(norm_param).shape:
                    model.layers[layer_idx].att_norm.__getattr__(norm_param).data.copy_(old_param)
        
        # FFN权重转移
        old_key = f'bert.encoder.layer.{layer_idx}.intermediate.dense.weight'
        if old_key in state_dict:
            old_weight = state_dict[old_key]  # shape: [3072, 768]
            new_weight = old_weight[keep_neurons]  # 选择保留的神经元行
            
            if new_weight.shape == model.layers[layer_idx].intermediate.weight.shape:
                model.layers[layer_idx].intermediate.weight.data.copy_(new_weight)
        
        old_key = f'bert.encoder.layer.{layer_idx}.intermediate.dense.bias'
        if old_key in state_dict:
            old_bias = state_dict[old_key]  # shape: [3072]
            new_bias = old_bias[keep_neurons]  # 选择保留的神经元
            
            if new_bias.shape == model.layers[layer_idx].intermediate.bias.shape:
                model.layers[layer_idx].intermediate.bias.data.copy_(new_bias)
        
        old_key = f'bert.encoder.layer.{layer_idx}.output.dense.weight'
        if old_key in state_dict:
            old_weight = state_dict[old_key]  # shape: [768, 3072]
            new_weight = old_weight[:, keep_neurons]  # 选择保留的神经元列
            
            if new_weight.shape == model.layers[layer_idx].output.weight.shape:
                model.layers[layer_idx].output.weight.data.copy_(new_weight)
        
        old_key = f'bert.encoder.layer.{layer_idx}.output.dense.bias'
        if old_key in state_dict:
            old_bias = state_dict[old_key]
            if old_bias.shape == model.layers[layer_idx].output.bias.shape:
                model.layers[layer_idx].output.bias.data.copy_(old_bias)
        
        # FFN LayerNorm转移
        for norm_param in ['weight', 'bias']:
            old_key = f'bert.encoder.layer.{layer_idx}.output.LayerNorm.{norm_param}'
            if old_key in state_dict:
                old_param = state_dict[old_key]
                if old_param.shape == model.layers[layer_idx].output_norm.__getattr__(norm_param).shape:
                    model.layers[layer_idx].output_norm.__getattr__(norm_param).data.copy_(old_param)
    
    # 转移pooler权重
    pooler_keys = ['bert.pooler.dense.weight', 'bert.pooler.dense.bias']
    pooler_targets = ['pooler.weight', 'pooler.bias']
    for old_key, new_key in zip(pooler_keys, pooler_targets):
        if old_key in state_dict and new_key in model.state_dict():
            model.state_dict()[new_key].copy_(state_dict[old_key])
            print(f"转移权重: {old_key} -> {new_key}")
    
    # 转移分类器权重
    classifier_keys = ['classifier.weight', 'classifier.bias']
    for key in classifier_keys:
        if key in state_dict and key in model.state_dict():
            model.state_dict()[key].copy_(state_dict[key])
            print(f"转移权重: {key}")
    
    print("权重转移完成!")


def getPrunedModel():
    try:
        # 修复4：检查文件是否存在
        head_mask_path = "outputs/bert-base-uncased/qnli/mac/0.5/seed_0/head_mask.pt"
        neuron_mask_path = "outputs/bert-base-uncased/qnli/mac/0.5/seed_0/neuron_mask.pt"
        model_path = "pretrained/qnli/pytorch_model.bin"
        
        print("加载mask文件...")
        head_mask = torch.load(head_mask_path).bool()
        neuron_mask = torch.load(neuron_mask_path).bool()
        
        print(f"Head mask shape: {head_mask.shape}")
        print(f"Neuron mask shape: {neuron_mask.shape}")

        config = BertConfig.from_pretrained("bert-base-uncased", num_labels=2)
        print("创建剪枝模型...")
        model = PrunedBertForSequenceClassification(config, head_mask, neuron_mask)

        print("加载权重文件...")
        state_dict = torch.load(model_path, map_location='cpu')
        
        # 修复5：使用自定义权重转移函数而不是直接load_state_dict
        load_and_transfer_weights(model, state_dict, head_mask, neuron_mask)
        
        # 验证模型结构
        print("\n模型结构验证:")
        for i, layer in enumerate(model.layers):
            print(f"Layer {i} - Num Attention Heads: {layer.num_heads}")
            print(f"Layer {i} - FFN Hidden Units: {layer.intermediate.out_features}")
        
        # 修复6：简单的前向传播测试
        print("\n测试前向传播...")
        test_input = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10
        try:
            with torch.no_grad():
                output = model(test_input)
            print(f"前向传播成功! 输出形状: {output.shape}")
        except Exception as e:
            print(f"前向传播失败: {str(e)}")
        
        return model
        
    except FileNotFoundError as e:
        print(f"文件未找到: {str(e)}")
        print("请确保所有必要的文件都存在")
        return None
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    model = getPrunedModel()