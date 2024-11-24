# BlaGPT

A collection of LM architectures, layers, and tricks to easily benchmark on a relatively small dataset. It is created fully for my experiments in my free time with no serious intentions. 

## BlaGPT Model
BlaGPT is a flexible Transformer implementation that you can turn on/off following things in the config. 

Multi-token prediction - [link](https://arxiv.org/pdf/2404.19737)

Weight tying - [link](https://arxiv.org/abs/1608.05859v3)

Grouped query attention - [link](https://arxiv.org/pdf/2305.13245)

Capping logits - [link](https://arxiv.org/pdf/2408.00118)

QKV bias - [link](https://arxiv.org/abs/2407.10671)

Zero-init projection layer - [link](https://arxiv.org/abs/2407.10671)

Post and pre-RMSNorm - [link](https://arxiv.org/pdf/2408.00118)

Tokenformer - [link](https://github.com/Haiyang-W/TokenFormer)

## Other Models
MegaByte - [link](https://arxiv.org/abs/2305.07185)

FTP (heavily modified) - [link](https://arxiv.org/pdf/2410.18160)

Rene - [link](https://huggingface.co/cartesia-ai/Rene-v0.1-1.3b-pytorch)

Rwkv7 - [link](https://github.com/BlinkDL/RWKV-LM)

Zamba2 - [link](https://huggingface.co/Zyphra/Zamba2-2.7B) - Zamba2 > Rene > Rwkv7

Hourglass Transformer (modified) - [link](https://arxiv.org/abs/2110.13711) - Hourglass > MegaByte > FTP

## Optimizers
PaLMForeachSOAP - [link](https://github.com/ClashLuke/HeavyBall) - almost 2 times slower than Adam but the best results

Ademamix - [link](https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch/blob/main/AdEMAMix.py) - Unstable even after trying different learning rates.

Adopt - [link](https://github.com/iShohei220/adopt) - straight up Nan

## Adding a New Model 

- Implement the model
- Return the loss in the forward function
- Register the model
- And start training

See one of the implementations for details. 


## Training

```bash
torchrun --standalone --nproc_per_node=8 train.py --run_name pre_post_norm --model_name blagpt
```

## Acknowledgements

The initial code is based on

Nano GPT - [link](https://github.com/karpathy/nanoGPT)

Modded NanoGPT - [link](https://github.com/KellerJordan/modded-nanogpt)
