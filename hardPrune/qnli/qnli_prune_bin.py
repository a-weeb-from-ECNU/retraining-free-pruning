import torch

def prune_bert_weights(original_state_dict, head_mask, neuron_mask, config):
    """
    original_state_dict: 加载的原始模型权重，cpu tensor
    head_mask: [num_layers, num_heads], bool tensor或0/1 mask，cpu tensor
    neuron_mask: [num_layers, intermediate_size], bool tensor或0/1 mask，cpu tensor
    config: bert config，获取num_layers, hidden_size, num_heads等信息
    """

    pruned_state_dict = {}

    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads
    intermediate_size = config.intermediate_size

    for key, weight in original_state_dict.items():
        if key.startswith("bert.encoder.layer."):
            parts = key.split(".")
            layer_idx = int(parts[3])

            if "attention.self.query.weight" in key or "attention.self.key.weight" in key or "attention.self.value.weight" in key:
                weight = weight.view(num_heads, head_dim, hidden_size)
                keep_heads = head_mask[layer_idx].nonzero(as_tuple=False).flatten()
                pruned_weight = weight[keep_heads.cpu()]  # 设备保持cpu
                pruned_weight = pruned_weight.reshape(-1, hidden_size)
                pruned_state_dict[key] = pruned_weight

            elif "attention.self.query.bias" in key or "attention.self.key.bias" in key or "attention.self.value.bias" in key:
                weight = weight.view(num_heads, head_dim)
                keep_heads = head_mask[layer_idx].nonzero(as_tuple=False).flatten()
                pruned_weight = weight[keep_heads.cpu()]
                pruned_weight = pruned_weight.reshape(-1)
                pruned_state_dict[key] = pruned_weight

            elif "attention.output.dense.weight" in key:
                weight = weight.view(hidden_size, num_heads, head_dim)
                keep_heads = head_mask[layer_idx].nonzero(as_tuple=False).flatten()
                pruned_weight = weight[:, keep_heads.cpu(), :]
                pruned_weight = pruned_weight.reshape(hidden_size, -1)
                pruned_state_dict[key] = pruned_weight

            elif "attention.output.dense.bias" in key:
                pruned_state_dict[key] = weight

            elif "intermediate.dense.weight" in key:
                keep_neurons = neuron_mask[layer_idx].nonzero(as_tuple=False).flatten()
                pruned_weight = weight[keep_neurons.cpu(), :]
                pruned_state_dict[key] = pruned_weight

            elif "intermediate.dense.bias" in key:
                keep_neurons = neuron_mask[layer_idx].nonzero(as_tuple=False).flatten()
                pruned_weight = weight[keep_neurons.cpu()]
                pruned_state_dict[key] = pruned_weight

            elif "output.dense.weight" in key:
                keep_neurons = neuron_mask[layer_idx].nonzero(as_tuple=False).flatten()
                pruned_weight = weight[:, keep_neurons.cpu()]
                pruned_state_dict[key] = pruned_weight

            elif "output.dense.bias" in key:
                pruned_state_dict[key] = weight

            else:
                pruned_state_dict[key] = weight

        else:
            pruned_state_dict[key] = weight

    return pruned_state_dict


def main():
    device = torch.device("cpu")

    original_state_dict = torch.load("pretrained/qnli/pytorch_model.bin", map_location=device)
    head_mask = torch.load("outputs/bert-base-uncased/qnli/mac/0.5/seed_0/head_mask.pt").to(device)
    neuron_mask = torch.load("outputs/bert-base-uncased/qnli/mac/0.5/seed_0/neuron_mask.pt").to(device)

    from transformers import BertConfig
    config = BertConfig.from_pretrained("bert-base-uncased")

    pruned_state_dict = prune_bert_weights(original_state_dict, head_mask, neuron_mask, config)

    torch.save(pruned_state_dict, "pytorch_model_pruned.bin")
    print("剪枝权重保存完毕: pytorch_model_pruned.bin")


if __name__ == "__main__":
    main()
