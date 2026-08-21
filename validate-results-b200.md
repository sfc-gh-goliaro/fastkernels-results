
THROUGHPUT / E2E WORKLOADS
MODEL                                                                         WORKLOAD                    REFERENCE                           SPEEDUP  CORRECTNESS
----------------------------------------------------------------------------  --------------------------  ----------------------------------  -------  -----------------------------------------------------------
meta-llama/Llama-3.1-8B-Instruct                                              mixed                       vLLM                                0.95x    avg prefix match len: 144.2; exact match: 291/1000 (29.1%)
meta-llama/Llama-3.1-8B-Instruct                                              long-context                vLLM                                0.96x    avg prefix match len: 128.4; exact match: 18/64 (28.1%)
mistralai/Mixtral-8x7B-Instruct-v0.1                                          mixed                       vLLM                                1.16x    avg prefix match len: 89.3; exact match: 150/1000 (15.0%)
mistralai/Mixtral-8x7B-Instruct-v0.1                                          long-context                vLLM                                0.97x    avg prefix match len: 83.4; exact match: 6/64 (9.4%)
nvidia/GLM-5.2-NVFP4                                                          mixed                       vLLM                                2.02x    avg prefix match len: 25.8; exact match: 53/1000 (5.3%)
nvidia/GLM-5.2-NVFP4                                                          long-context                vLLM                                0.98x    avg prefix match len: 41.2; exact match: 8/64 (12.5%)
microsoft/bitnet-b1.58-2B-4T                                                  mixed                       microsoft-bitnet-gpu                1.20x    avg prefix match len: 202.8; exact match: 6/32 (18.8%)
microsoft/bitnet-b1.58-2B-4T                                                  long-context                microsoft-bitnet-gpu                1.01x    avg prefix match len: 41.4; exact match: 1/32 (3.1%)
openai/gpt-oss-120b                                                           mixed                       vLLM                                0.97x    avg prefix match len: 57.3; exact match: 133/1000 (13.3%)
openai/gpt-oss-120b                                                           long-context                vLLM                                0.93x    avg prefix match len: 50.3; exact match: 2/64 (3.1%)
meta-llama/Llama-3.1-8B-Instruct                                              eagle3-16seqs-out256        SGLang                              0.93x    avg prefix match len: 160.6; exact match: 6/16 (37.5%)
google/gemma-4-26B-A4B-it                                                     mixed                       vLLM                                1.00x    avg prefix match len: 67.9; exact match: 144/1000 (14.4%)
google/gemma-4-26B-A4B-it                                                     long-context                vLLM                                1.00x    avg prefix match len: 72.0; exact match: 3/64 (4.7%)
state-spaces/mamba-2.8b-hf                                                    mixed                       vLLM                                3.56x    avg prefix match len: 166.0; exact match: 372/1000 (37.2%)
state-spaces/mamba-2.8b-hf                                                    long-context                vLLM                                1.24x    avg prefix match len: 93.0; exact match: 17/64 (26.6%)
mistralai/Mamba-Codestral-7B-v0.1                                             mixed                       vLLM                                0.96x    avg prefix match len: 206.1; exact match: 437/1000 (43.7%)
mistralai/Mamba-Codestral-7B-v0.1                                             long-context                vLLM                                0.97x    avg prefix match len: 200.1; exact match: 42/64 (65.6%)
fla-hub/rwkv7-2.9B-g1                                                         mixed                       FLA                                 1.27x    avg prefix match len: 96.5; exact match: 172/1000 (17.2%)
fla-hub/rwkv7-2.9B-g1                                                         long-context                FLA                                 1.18x    avg prefix match len: 104.9; exact match: 8/64 (12.5%)
fla-hub/gla-2.7B-100B                                                         mixed                       FLA                                 1.25x    avg prefix match len: 332.3; exact match: 730/1000 (73.0%)
fla-hub/gla-2.7B-100B                                                         long-context                FLA                                 1.65x    avg prefix match len: 192.0; exact match: 44/64 (68.8%)
fla-hub/retnet-2.7B-100B                                                      mixed                       FLA                                 6.83x    avg prefix match len: 21.2; exact match: 50/1000 (5.0%)
fla-hub/retnet-2.7B-100B                                                      long-context                FLA                                 1.69x    avg prefix match len: 167.6; exact match: 35/64 (54.7%)
Qwen/Qwen3-Next-80B-A3B-Instruct                                              mixed                       vLLM                                1.46x    avg prefix match len: 9.8; exact match: 28/1000 (2.8%)
Qwen/Qwen3-Next-80B-A3B-Instruct                                              long-context                vLLM                                1.01x    avg prefix match len: 13.6; exact match: 0/64 (0.0%)
moonshotai/Kimi-Linear-48B-A3B-Instruct                                       mixed                       vLLM                                1.51x    avg prefix match len: 41.8; exact match: 107/1000 (10.7%)
moonshotai/Kimi-Linear-48B-A3B-Instruct                                       long-context                vLLM                                0.97x    avg prefix match len: 85.7; exact match: 5/64 (7.8%)
ttt_e2e                                                                       ttt-e2e-pretrain            jax                                 3.64x    pass=True; token_nll_mean_abs_diff=0.022682
ttt_e2e                                                                       ttt-e2e-meta                jax                                 1.05x    pass=True; token_nll_mean_abs_diff=0.021899
ai21labs/AI21-Jamba-Mini-1.7                                                  mixed                       vLLM                                0.95x    avg prefix match len: 82.5; exact match: 155/1000 (15.5%)
ai21labs/AI21-Jamba-Mini-1.7                                                  long-context                vLLM                                0.74x    avg prefix match len: 100.3; exact match: 10/64 (15.6%)
black-forest-labs/FLUX.1-dev                                                  1024x1024                   vllm-omni                           1.04x    min_cos=0.9493
black-forest-labs/FLUX.1-dev                                                  512x512                     vllm-omni                           1.66x    min_cos=0.9849
hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v                    480p-short                  vllm-omni                           1.03x    min_cos=0.9588
hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v                    480p-medium                 vllm-omni                           1.00x    min_cos=0.9739
stabilityai/stable-diffusion-xl-base-1.0                                      1024x1024                   diffusers                           1.02x    min_cos=0.9623
stabilityai/stable-diffusion-xl-base-1.0                                      512x512                     diffusers                           1.04x    min_cos=0.9193
facebook/sam3.1                                                               full-pipeline               sam3                                1.03x    pass=True; min_cosine=0.871415
facebook/sam3.1                                                               smartglasses-val-video      sam3                                1.06x    min_cosine=0.928175
openai/whisper-large-v3                                                       librispeech                 vLLM                                1.07x    avg prefix match len: 385.9; exact match: 1664/2620 (63.5%)
FunAudioLLM/Fun-CosyVoice3-0.5B-2512                                          tts-short                   vllm-omni                           2.09x    -
FunAudioLLM/Fun-CosyVoice3-0.5B-2512                                          tts-medium                  vllm-omni                           2.38x    -
FunAudioLLM/Fun-CosyVoice3-0.5B-2512                                          tts-long                    vllm-omni                           2.44x    -
Qwen/Qwen2-VL-7B-Instruct                                                     text-only                   vLLM                                1.01x    avg prefix match len: 169.5; exact match: 364/1000 (36.4%)
Qwen/Qwen2-VL-7B-Instruct                                                     image                       vLLM                                1.25x    avg prefix match len: 255.7; exact match: 372/1000 (37.2%)
Qwen/Qwen2-VL-7B-Instruct                                                     video                       vLLM                                1.50x    avg prefix match len: 312.7; exact match: 582/1000 (58.2%)
Qwen/Qwen3-VL-8B-Instruct                                                     text-only                   vLLM                                1.00x    avg prefix match len: 97.4; exact match: 197/1000 (19.7%)
Qwen/Qwen3-VL-8B-Instruct                                                     image                       vLLM                                1.22x    avg prefix match len: 115.5; exact match: 76/1000 (7.6%)
Qwen/Qwen3-VL-8B-Instruct                                                     video                       vLLM                                1.65x    avg prefix match len: 210.4; exact match: 332/1000 (33.2%)
Qwen/Qwen3-VL-235B-A22B-Instruct-FP8                                          text-only                   vLLM                                1.32x    avg prefix match len: 36.7; exact match: 78/1000 (7.8%)
Qwen/Qwen3-VL-235B-A22B-Instruct-FP8                                          image                       vLLM                                2.04x    avg prefix match len: 49.5; exact match: 8/1000 (0.8%)
Qwen/Qwen3-VL-235B-A22B-Instruct-FP8                                          video                       vLLM                                1.49x    avg prefix match len: 34.0; exact match: 2/1000 (0.2%)
Qwen/Qwen2.5-Omni-7B                                                          text                        vLLM                                0.99x    avg prefix match len: 161.1; exact match: 332/1000 (33.2%)
Qwen/Qwen2.5-Omni-7B                                                          image                       vLLM                                1.29x    avg prefix match len: 144.3; exact match: 133/1000 (13.3%)
Qwen/Qwen2.5-Omni-7B                                                          video                       vLLM                                1.59x    avg prefix match len: 190.3; exact match: 233/1000 (23.3%)
Qwen/Qwen2.5-Omni-7B                                                          audio                       vLLM                                1.80x    avg prefix match len: 150.1; exact match: 376/1000 (37.6%)
google/siglip2-so400m-patch16-naflex                                          default-res                 timm                                1.01x    min_cos=0.9997
google/siglip2-so400m-patch16-naflex                                          high-res                    timm                                1.01x    min_cos=1.0000
facebook/dinov3-vit7b16-pretrain-lvd1689m                                     default-res                 timm                                0.94x    min_cos=1.0000
facebook/dinov3-vit7b16-pretrain-lvd1689m                                     high-res                    timm                                0.94x    min_cos=1.0000
timm/swinv2_large_window12_192.ms_in22k                                       default-res                 timm                                0.94x    min_cos=1.0000
timm/swinv2_large_window12_192.ms_in22k                                       high-res                    timm                                0.95x    min_cos=1.0000
timm/mobilenetv4_conv_medium.e500_r256_in1k                                   default-res                 timm                                1.07x    min_cos=1.0000
timm/mobilenetv4_conv_medium.e500_r256_in1k                                   high-res                    timm                                0.99x    min_cos=1.0000
facebook/convnextv2-base-22k-384                                              imagecls-ethz/food101       transformers                        1.00x    pass=True; top1_match_rate=1.000000
timm/efficientnetv2_rw_m.agc_in1k                                             imagecls-ethz/food101       timm                                1.04x    pass=True; top1_match_rate=1.000000
jameslahm/yolov10n                                                            coco-val                    reference                           1.14x    boxes=0.9973; scores=0.9999; labels=100.0%
PekingU/rtdetr_v2_r101vd                                                      coco-val                    reference                           0.97x    boxes=0.9277; scores=1.0000; labels=73.4%
Voxel51/gaussian_splatting                                                    render                      gsplat                              1.00x    pass=True; rgb_cosine=1.000000
NVlabs/instant-ngp                                                            render                      pyngp                               0.98x    pass=True; rgba_cosine=1.000000
Pointcept/PointTransformerV3                                                  scanobjectnn                official-detached                   0.97x    pass=False; feat_cosine=0.986230
Pointcept/PointTransformerV3                                                  batch-8                     official-detached                   1.00x    pass=False; feat_cosine=0.986230
OpenFold/OpenFold3                                                            short                       reference                           1.00x    align=100.0%
OpenFold/OpenFold3                                                            medium                      reference                           1.02x    align=100.0%
OpenFold/OpenFold3                                                            long                        reference                           1.00x    align=100.0%
OpenFold/OpenFold3                                                            extra-long                  reference                           1.01x    align=100.0%
/home/yak/.fastkernels/third_party/openpi-assets/pi0_aloha_pen_uncap_pytorch  aloha-3cam                  openpi                              3.93x    pass=True; mean_cosine_sim=0.999998
/home/yak/.fastkernels/third_party/openpi-assets/pi0_aloha_pen_uncap_pytorch  aloha-1cam                  openpi                              2.43x    pass=True; mean_cosine_sim=0.999994
dp3                                                                           dp3-1env                    3D-Diffusion-Policy                 1.13x    min_cos=1.0000
dp3                                                                           dp3-batch                   3D-Diffusion-Policy                 1.14x    min_cos=1.0000
dlrmv2                                                                        ctr-batch                   torchrec.models.dlrm.DLRM           1.10x    pass=False; min_cosine=nan
lightgcn                                                                      recommend-batch             torch_geometric.nn.models.LightGCN  1.02x    pass=True; min_cosine=1.000000
BAAI/bge-m3                                                                   bge-m3-mldr-docs            vLLM                                2.35x    pass=True; min_cos=0.998640
colbert-ir/colbertv2.0                                                        colbertv2-msmarco-passages  vLLM                                3.13x    pass=True; min_cos=0.999946
GSAI-ML/LLaDA-8B-Instruct                                                     humaneval-fastdllm-dual     fastdllm-dual                       0.96x    avg prefix match len: 168.6; exact match: 92/164 (56.1%)
Etched/oasis-500m                                                             latency-bs1-8f-4ddim        open-oasis                          1.16x    -
Etched/oasis-500m                                                             short-bs4-16f-4ddim         open-oasis                          1.14x    pass=True; min_cos=0.9985
Etched/oasis-500m                                                             medium-bs8-24f-4ddim        open-oasis                          1.19x    pass=True; min_cos=0.9990
Etched/oasis-500m                                                             long-bs8-32f-4ddim          open-oasis                          1.19x    pass=True; min_cos=0.9979
Etched/oasis-500m                                                             denoise-bs4-16f-8ddim       open-oasis                          1.18x    pass=True; min_cos=0.9974
facebook/vjepa2-vitl-fpc64-256                                                predictor                   transformers                        0.97x    pass=True; min_cosine=1.000000
facebook/vjepa2-vitl-fpc64-256                                                encoder                     transformers                        0.98x    pass=True; min_cosine=1.000000

LATENCY WORKLOADS
MODEL                                                                         WORKLOAD                         REFERENCE                           SPEEDUP
----------------------------------------------------------------------------  -------------------------------  ----------------------------------  -------
meta-llama/Llama-3.1-8B-Instruct                                              single-request                   vLLM                                1.03x
meta-llama/Llama-3.1-8B-Instruct                                              fixed-batch-32                   vLLM                                1.02x
mistralai/Mixtral-8x7B-Instruct-v0.1                                          single-request                   vLLM                                1.00x
mistralai/Mixtral-8x7B-Instruct-v0.1                                          fixed-batch-32                   vLLM                                1.00x
nvidia/GLM-5.2-NVFP4                                                          single-request                   vLLM                                0.95x
nvidia/GLM-5.2-NVFP4                                                          fixed-batch-32                   vLLM                                0.93x
microsoft/bitnet-b1.58-2B-4T                                                  single-request                   microsoft-bitnet-gpu                1.20x
openai/gpt-oss-120b                                                           single-request                   vLLM                                0.91x
openai/gpt-oss-120b                                                           fixed-batch-32                   vLLM                                0.99x
meta-llama/Llama-3.1-8B-Instruct                                              latency-bs1-out256               SGLang                              1.04x
google/gemma-4-26B-A4B-it                                                     single-request                   vLLM                                1.06x
google/gemma-4-26B-A4B-it                                                     fixed-batch-32                   vLLM                                1.12x
state-spaces/mamba-2.8b-hf                                                    single-request                   vLLM                                2.77x
state-spaces/mamba-2.8b-hf                                                    fixed-batch-32                   vLLM                                1.47x
mistralai/Mamba-Codestral-7B-v0.1                                             single-request                   vLLM                                0.95x
mistralai/Mamba-Codestral-7B-v0.1                                             fixed-batch-32                   vLLM                                0.95x
fla-hub/rwkv7-2.9B-g1                                                         single-request                   FLA                                 1.22x
fla-hub/rwkv7-2.9B-g1                                                         fixed-batch-32                   FLA                                 1.30x
fla-hub/gla-2.7B-100B                                                         single-request                   FLA                                 1.82x
fla-hub/gla-2.7B-100B                                                         fixed-batch-32                   FLA                                 1.68x
fla-hub/retnet-2.7B-100B                                                      single-request                   FLA                                 1.74x
fla-hub/retnet-2.7B-100B                                                      fixed-batch-32                   FLA                                 1.72x
Qwen/Qwen3-Next-80B-A3B-Instruct                                              single-request                   vLLM                                0.99x
Qwen/Qwen3-Next-80B-A3B-Instruct                                              fixed-batch-32                   vLLM                                0.99x
moonshotai/Kimi-Linear-48B-A3B-Instruct                                       single-request                   vLLM                                1.07x
moonshotai/Kimi-Linear-48B-A3B-Instruct                                       fixed-batch-32                   vLLM                                1.02x
ttt_e2e                                                                       ttt-e2e-pretrain-forward         jax                                 3.64x
ttt_e2e                                                                       ttt-e2e-meta-forward             jax                                 1.05x
ai21labs/AI21-Jamba-Mini-1.7                                                  single-request                   vLLM                                1.18x
ai21labs/AI21-Jamba-Mini-1.7                                                  fixed-batch-32                   vLLM                                1.03x
black-forest-labs/FLUX.1-dev                                                  single-1024x1024                 vllm-omni                           1.00x
black-forest-labs/FLUX.1-dev                                                  single-512x512                   vllm-omni                           1.21x
hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v                    single-480p-short                vllm-omni                           1.10x
hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v                    single-480p-medium               vllm-omni                           1.13x
stabilityai/stable-diffusion-xl-base-1.0                                      single-1024x1024                 diffusers                           1.03x
stabilityai/stable-diffusion-xl-base-1.0                                      single-512x512                   diffusers                           1.02x
facebook/sam3.1                                                               single-image-1008                sam3                                1.06x
facebook/sam3.1                                                               batch-4-image-1008               sam3                                1.06x
facebook/sam3.1                                                               single-video-frame-1008          sam3                                1.07x
openai/whisper-large-v3                                                       single-utterance                 vLLM                                1.03x
openai/whisper-large-v3                                                       fixed-batch-32                   vLLM                                1.04x
FunAudioLLM/Fun-CosyVoice3-0.5B-2512                                          single-utterance                 vllm-omni                           2.34x
Qwen/Qwen2-VL-7B-Instruct                                                     single-image                     vLLM                                1.03x
Qwen/Qwen2-VL-7B-Instruct                                                     single-video                     vLLM                                0.99x
Qwen/Qwen3-VL-8B-Instruct                                                     single-image                     vLLM                                1.06x
Qwen/Qwen3-VL-8B-Instruct                                                     single-video                     vLLM                                1.05x
Qwen/Qwen3-VL-235B-A22B-Instruct-FP8                                          single-image                     vLLM                                0.93x
Qwen/Qwen3-VL-235B-A22B-Instruct-FP8                                          single-video                     vLLM                                0.98x
Qwen/Qwen2.5-Omni-7B                                                          single-text                      vLLM                                1.03x
Qwen/Qwen2.5-Omni-7B                                                          single-image                     vLLM                                1.07x
Qwen/Qwen2.5-Omni-7B                                                          single-video                     vLLM                                1.13x
Qwen/Qwen2.5-Omni-7B                                                          single-audio                     vLLM                                1.07x
google/siglip2-so400m-patch16-naflex                                          single-image                     timm                                1.11x
google/siglip2-so400m-patch16-naflex                                          batch-8                          timm                                1.01x
facebook/dinov3-vit7b16-pretrain-lvd1689m                                     single-image                     timm                                0.94x
facebook/dinov3-vit7b16-pretrain-lvd1689m                                     batch-8                          timm                                0.94x
timm/swinv2_large_window12_192.ms_in22k                                       single-image                     timm                                0.92x
timm/swinv2_large_window12_192.ms_in22k                                       batch-8                          timm                                0.90x
timm/mobilenetv4_conv_medium.e500_r256_in1k                                   single-image                     timm                                1.05x
timm/mobilenetv4_conv_medium.e500_r256_in1k                                   batch-8                          timm                                1.04x
facebook/convnextv2-base-22k-384                                              batch-1                          transformers                        1.08x
facebook/convnextv2-base-22k-384                                              batch-8                          transformers                        1.00x
timm/efficientnetv2_rw_m.agc_in1k                                             batch-1                          timm                                1.09x
timm/efficientnetv2_rw_m.agc_in1k                                             batch-8                          timm                                1.09x
jameslahm/yolov10n                                                            single-image                     reference                           0.94x
jameslahm/yolov10n                                                            batch-4                          reference                           0.95x
PekingU/rtdetr_v2_r101vd                                                      single-image                     reference                           0.91x
PekingU/rtdetr_v2_r101vd                                                      batch-4                          reference                           0.89x
Voxel51/gaussian_splatting                                                    single-render                    gsplat                              0.99x
NVlabs/instant-ngp                                                            single-render                    pyngp                               0.98x
Pointcept/PointTransformerV3                                                  single-cloud                     official-detached                   0.99x
OpenFold/OpenFold3                                                            single-short                     reference                           1.05x
OpenFold/OpenFold3                                                            single-medium                    reference                           1.00x
OpenFold/OpenFold3                                                            single-long                      reference                           1.03x
OpenFold/OpenFold3                                                            single-extra-long                reference                           1.02x
/home/yak/.fastkernels/third_party/openpi-assets/pi0_aloha_pen_uncap_pytorch  aloha-single-3cam                openpi                              3.90x
/home/yak/.fastkernels/third_party/openpi-assets/pi0_aloha_pen_uncap_pytorch  aloha-single-1cam                openpi                              4.46x
dp3                                                                           single-step                      3D-Diffusion-Policy                 1.14x
dp3                                                                           batch-8                          3D-Diffusion-Policy                 1.12x
dlrmv2                                                                        single-request                   torchrec.models.dlrm.DLRM           1.41x
dlrmv2                                                                        fixed-batch-32                   torchrec.models.dlrm.DLRM           1.33x
lightgcn                                                                      single-request                   torch_geometric.nn.models.LightGCN  1.04x
lightgcn                                                                      fixed-batch-32                   torch_geometric.nn.models.LightGCN  1.04x
BAAI/bge-m3                                                                   single-request                   vLLM                                0.98x
BAAI/bge-m3                                                                   fixed-batch-32                   vLLM                                0.97x
colbert-ir/colbertv2.0                                                        single-request                   vLLM                                0.78x
colbert-ir/colbertv2.0                                                        fixed-batch-32                   vLLM                                2.26x
GSAI-ML/LLaDA-8B-Instruct                                                     humaneval-fastdllm-dual-batch32  fastdllm-dual                       0.96x
Etched/oasis-500m                                                             latency-bs1-8f-4ddim             open-oasis                          1.16x
facebook/vjepa2-vitl-fpc64-256                                                single-video                     transformers                        0.98x