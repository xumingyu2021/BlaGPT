# BlaGPT

Code base started from:

Nano GPT - [link](https://github.com/karpathy/nanoGPT)
Modded NanoGPT - [link](https://github.com/KellerJordan/modded-nanogpt)

## BlaGPT Model
Multi token prediction - [link](https://arxiv.org/pdf/2404.19737)
Weight tying - [link](https://arxiv.org/abs/1608.05859v3)
Grouped query attention - [link](https://arxiv.org/pdf/2305.13245)
Capping logits - [link](https://arxiv.org/pdf/2408.00118)
QKV bias - [link](https://arxiv.org/abs/2407.10671)
Zero-init projection layer - [link](https://arxiv.org/abs/2407.10671)
Post and pre RMSNorm - [link](https://arxiv.org/pdf/2408.00118)

## Other Models
MegaByte - [link](https://arxiv.org/abs/2305.07185)
FTP (heavily modified) - [link](https://arxiv.org/pdf/2410.18160)
Rene - [link](https://huggingface.co/cartesia-ai/Rene-v0.1-1.3b-pytorch)
Rwkv7 - [link](https://github.com/BlinkDL/RWKV-LM)
Zamba2 - [link](https://huggingface.co/Zyphra/Zamba2-2.7B)
Hourglas Transformer (modified) - [link](https://arxiv.org/abs/2110.13711)


## Training

```bash
torchrun --standalone --nproc_per_node=8 train.py --run_name pre_post_norm --model_name blagpt
```