import torch


def softmax(x, dim=-1):
    original_dtype = x.dtype
    if original_dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return (exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)).to(original_dtype)
