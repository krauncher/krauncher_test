## krauncher

33 tasks; versions: {"calibration_id": "c-50c95de799e3", "client_commit": "0e2e207275f808b0e86fe9f0d49122ec993fe2d4"}

### Forecast time

median 0.63 s, p90 1.19 s, max 2.37 s

### Classification / assay fields

| field | match | differs in |
|---|---|---|
| workload_type | 31/33 | lambdalabs/ncf_ml20m_bs10m, lambdalabs/ncf_ml20m_bs4278184 |
| mode | 33/33 | - |
| framework | 33/33 | - |
| precision | 33/33 | - |
| params_billions | 25/33 | lambdalabs/ncf_ml20m_bs10m, lambdalabs/ncf_ml20m_bs4278184, lambdalabs/transformer_xl_base_wt103_bs104, lambdalabs/transformer_xl_base_wt103_bs24, lambdalabs/transformer_xl_base_wt103_bs64, lambdalabs/transformer_xl_large_wt103_bs32, lambdalabs/transformer_xl_large_wt103_bs48, lambdalabs/transformer_xl_large_wt103_bs8 |
| batch_size | 28/33 | krauncher_tutorials/phi3_inference, krauncher_tutorials/qwen15_inference, krauncher_tutorials/qwen25_7b_batched, krauncher_tutorials/qwen25_7b_gsm8k, krauncher_tutorials/qwen25_7b_long |
| epochs | 14/14 | - |
| dataset_samples | 13/22 | krauncher_tutorials/resnet152_food101, lambdalabs/ncf_ml20m_bs10m, lambdalabs/ncf_ml20m_bs4278184, lambdalabs/tacotron2_ljs625_bs148, lambdalabs/tacotron2_ljs625_bs256, lambdalabs/tacotron2_ljs625_bs88, lambdalabs/waveglow_ljs625_bs18, lambdalabs/waveglow_ljs625_bs32, lambdalabs/waveglow_ljs625_bs48 |
| seq_len | 15/15 | - |
| cpu_only | 33/33 | - |

### Measurements: krauncher-calibration-2026 (independent: False; 10 tasks)

#### Reference-card time

centre x0.943, typical deviation x1.065, inside the issued spread 8/10

| task | measured, s | forecast, s | forecast / measured | status |
|---|---|---|---|---|
| krauncher_tutorials/bert_batch_inference | 13 | 10.3 | x0.792 | critical |
| krauncher_tutorials/bert_imdb | 143 | 143.1 | x1.001 | good |
| krauncher_tutorials/phi3_inference | 133 | 139.4 | x1.048 | good |
| krauncher_tutorials/qwen15_inference | 77 | 75.8 | x0.984 | good |
| krauncher_tutorials/qwen25_7b_batched | 136 | 145.2 | x1.068 | good |
| krauncher_tutorials/qwen25_7b_gsm8k | 711 | 639.1 | x0.899 | warning |
| krauncher_tutorials/qwen25_7b_long | 204 | 199.1 | x0.976 | good |
| krauncher_tutorials/qwen25_7b_lora_alpaca | 622 | 661.0 | x1.063 | good |
| krauncher_tutorials/resnet152_food101 | 194 | 118.1 | x0.609 | critical |
| krauncher_tutorials/vit_batch_inference | 9 | 10.1 | x1.122 | warning |

#### VRAM

centre x1.108, typical deviation x1.128, below the measured peak 1/10

| task | measured, GB | forecast, GB | forecast / measured | status |
|---|---|---|---|---|
| krauncher_tutorials/bert_batch_inference | 0.8 | 1 | x1.25 | good |
| krauncher_tutorials/bert_imdb | 3.9 | 5 | x1.282 | good |
| krauncher_tutorials/phi3_inference | 8.8 | 9 | x1.023 | good |
| krauncher_tutorials/qwen15_inference | 4.0 | 4 | x1.0 | good |
| krauncher_tutorials/qwen25_7b_batched | 17.6 | 18 | x1.023 | good |
| krauncher_tutorials/qwen25_7b_gsm8k | 15.7 | 18 | x1.146 | good |
| krauncher_tutorials/qwen25_7b_long | 15.8 | 18 | x1.139 | good |
| krauncher_tutorials/qwen25_7b_lora_alpaca | 30.4 | 34 | x1.118 | good |
| krauncher_tutorials/resnet152_food101 | 13.2 | 12 | x0.909 | critical |
| krauncher_tutorials/vit_batch_inference | 0.8 | 1 | x1.25 | good |

#### Ladder (compute_ratio)

126 (task, GPU) pairs: typical error x1.263, p90 x2.012; median rank correlation 0.83; pick slower than the fastest in 3/10 tasks

| task | GPUs | typical error | p90 | rank corr | picked | fastest | time regret | status |
|---|---|---|---|---|---|---|---|---|
| krauncher_tutorials/bert_batch_inference | 10 | x1.309 | x1.721 | 0.85 | rtx_6000_blackwell | rtx_6000_blackwell | x1.0 | good |
| krauncher_tutorials/bert_imdb | 27 | x1.247 | x1.65 | 0.93 | b200 | b200 | x1.0 | good |
| krauncher_tutorials/phi3_inference | 10 | x1.335 | x2.012 | 0.84 | rtx_6000_blackwell | rtx_6000_blackwell | x1.0 | good |
| krauncher_tutorials/qwen15_inference | 10 | x1.365 | x2.826 | 0.66 | h100_sxm | rtx_6000_blackwell | x2.031 | critical |
| krauncher_tutorials/qwen25_7b_batched | 9 | x1.188 | x1.818 | 0.87 | rtx_6000_blackwell | rtx_6000_blackwell | x1.0 | good |
| krauncher_tutorials/qwen25_7b_gsm8k | 12 | x1.195 | x1.705 | 0.83 | rtx_6000_blackwell | rtx_6000_blackwell | x1.0 | good |
| krauncher_tutorials/qwen25_7b_long | 8 | x1.121 | x1.671 | 0.55 | rtx_6000_blackwell | rtx_6000_blackwell | x1.0 | warning |
| krauncher_tutorials/qwen25_7b_lora_alpaca | 10 | x1.215 | x1.696 | 0.92 | b300 | b300 | x1.0 | good |
| krauncher_tutorials/resnet152_food101 | 28 | x1.424 | x2.246 | 0.57 | b200 | rtx_6000_blackwell | x1.653 | critical |
| krauncher_tutorials/vit_batch_inference | 12 | x1.479 | x2.47 | 0.66 | h100_sxm | rtx_6000_blackwell | x1.538 | critical |

#### Ladder coverage

measured GPUs without a forecast: krauncher_tutorials/qwen25_7b_lora_alpaca 2/11

### Measurements: lambda-dlb-v1-fp16 (independent: True; 23 tasks)

#### Compute time on the anchor GPU

centre x1.027, typical deviation x3.279, within x1.1: 3/20

| task | GPU | measured, s | forecast, s | forecast / measured | status |
|---|---|---|---|---|---|
| lambdalabs/bert_base_squad_amp_bs192 | rtx_6000_ada | 75.3 | 20.4 | x0.271 | critical |
| lambdalabs/bert_base_squad_amp_bs320 | a100_sxm_80 | 72.2 | 20.8 | x0.288 | critical |
| lambdalabs/bert_base_squad_amp_bs96 | rtx_4090 | 32.3 | 22.8 | x0.705 | critical |
| lambdalabs/bert_large_squad_amp_bs112 | a100_sxm_80 | 73.7 | 20.8 | x0.282 | critical |
| lambdalabs/bert_large_squad_amp_bs28 | rtx_4090 | 30.1 | 22.8 | x0.757 | critical |
| lambdalabs/bert_large_squad_amp_bs64 | rtx_6000_ada | 81.0 | 20.4 | x0.252 | critical |
| lambdalabs/ncf_ml20m_bs10m | a100_sxm_80 | 5.8 | 2124.1 | x366.228 | critical |
| lambdalabs/ncf_ml20m_bs4278184 | rtx_4090 | 6.1 | 156.5 | x25.652 | critical |
| lambdalabs/resnet50_amp_bs1280 | a100_sxm_80 | 161.8 | 165.9 | x1.025 | good |
| lambdalabs/resnet50_amp_bs448 | rtx_4090 | 68.9 | 194.5 | x2.823 | critical |
| lambdalabs/resnet50_amp_bs928 | rtx_6000_ada | 40.5 | 44.4 | x1.096 | good |
| lambdalabs/transformer_xl_base_wt103_bs104 | a100_sxm_80 | 185.4 | 41.7 | x0.225 | critical |
| lambdalabs/transformer_xl_base_wt103_bs24 | rtx_4090 | 45.6 | 45.7 | x1.003 | good |
| lambdalabs/transformer_xl_base_wt103_bs64 | rtx_6000_ada | 13.4 | 4.1 | x0.306 | critical |
| lambdalabs/transformer_xl_large_wt103_bs32 | rtx_6000_ada | 20.4 | 2.1 | x0.101 | critical |
| lambdalabs/transformer_xl_large_wt103_bs48 | a100_sxm_80 | 274.3 | 83.3 | x0.304 | critical |
| lambdalabs/transformer_xl_large_wt103_bs8 | rtx_4090 | 56.5 | 38.6 | x0.684 | critical |
| lambdalabs/waveglow_ljs625_bs18 | rtx_4090 | 60.9 | 227.7 | x3.739 | critical |
| lambdalabs/waveglow_ljs625_bs32 | rtx_6000_ada | 62.0 | 114.7 | x1.85 | critical |
| lambdalabs/waveglow_ljs625_bs48 | a100_sxm_80 | 40.8 | 77.9 | x1.91 | critical |

#### VRAM upper bound

forecast above the measured bound in 0/23 tasks

#### Ladder (compute_ratio)

55 (task, GPU) pairs: typical error x1.234, p90 x1.691; median rank correlation 1.0; pick slower than the fastest in 0/23 tasks

| task | GPUs | typical error | p90 | rank corr | picked | fastest | time regret | status |
|---|---|---|---|---|---|---|---|---|
| lambdalabs/bert_base_squad_amp_bs192 | 3 | x1.617 | x1.71 | 1.0 | rtx_6000_ada | rtx_6000_ada | x1.0 | good |
| lambdalabs/bert_base_squad_amp_bs320 | 4 | x1.043 | x1.167 | 1.0 | h100_sxm | h100_sxm | x1.0 | good |
| lambdalabs/bert_base_squad_amp_bs96 | 2 | x1.489 | x1.489 | None | rtx_4090 | rtx_4090 | x1.0 | good |
| lambdalabs/bert_large_squad_amp_bs112 | 4 | x1.025 | x1.216 | 1.0 | h100_sxm | h100_sxm | x1.0 | good |
| lambdalabs/bert_large_squad_amp_bs28 | 3 | x1.397 | x1.438 | 0.5 | rtx_4090 | rtx_4090 | x1.0 | warning |
| lambdalabs/bert_large_squad_amp_bs64 | 3 | x1.791 | x1.891 | 1.0 | rtx_6000_ada | rtx_6000_ada | x1.0 | good |
| lambdalabs/ncf_ml20m_bs10m | 7 | x1.285 | x1.691 | 1.0 | h100_sxm | h100_sxm | x1.0 | good |
| lambdalabs/ncf_ml20m_bs4278184 | 3 | x1.615 | x1.876 | 0.5 | rtx_4090 | rtx_4090 | x1.0 | warning |
| lambdalabs/resnet50_amp_bs1280 | 4 | x1.098 | x1.219 | 1.0 | h100_sxm | h100_sxm | x1.0 | good |
| lambdalabs/resnet50_amp_bs448 | 3 | x1.435 | x1.449 | 0.5 | rtx_4090 | rtx_4090 | x1.0 | warning |
| lambdalabs/resnet50_amp_bs928 | 3 | x1.388 | x1.561 | 1.0 | rtx_6000_ada | rtx_6000_ada | x1.0 | good |
| lambdalabs/tacotron2_ljs625_bs148 | 3 | x1.138 | x1.22 | 1.0 | rtx_6000_ada | rtx_6000_ada | x1.0 | good |
| lambdalabs/tacotron2_ljs625_bs256 | 4 | x1.217 | x1.429 | 1.0 | h100_sxm | h100_sxm | x1.0 | good |
| lambdalabs/tacotron2_ljs625_bs88 | 2 | x1.629 | x1.629 | None | rtx_4090 | rtx_4090 | x1.0 | good |
| lambdalabs/transformer_xl_base_wt103_bs104 | 4 | x1.15 | x1.196 | 1.0 | h100_sxm | h100_sxm | x1.0 | good |
| lambdalabs/transformer_xl_base_wt103_bs24 | 3 | x1.488 | x1.523 | 0.5 | rtx_4090 | rtx_4090 | x1.0 | warning |
| lambdalabs/transformer_xl_base_wt103_bs64 | 3 | x1.438 | x1.482 | 1.0 | rtx_6000_ada | rtx_6000_ada | x1.0 | good |
| lambdalabs/transformer_xl_large_wt103_bs32 | 3 | x1.413 | x1.462 | 1.0 | rtx_6000_ada | rtx_6000_ada | x1.0 | good |
| lambdalabs/transformer_xl_large_wt103_bs48 | 4 | x1.045 | x1.061 | 1.0 | h100_sxm | h100_sxm | x1.0 | good |
| lambdalabs/transformer_xl_large_wt103_bs8 | 3 | x1.523 | x1.902 | 0.5 | rtx_4090 | rtx_4090 | x1.0 | warning |
| lambdalabs/waveglow_ljs625_bs18 | 3 | x1.38 | x1.553 | 0.5 | rtx_4090 | rtx_4090 | x1.0 | warning |
| lambdalabs/waveglow_ljs625_bs32 | 3 | x1.324 | x1.467 | 1.0 | rtx_6000_ada | rtx_6000_ada | x1.0 | good |
| lambdalabs/waveglow_ljs625_bs48 | 4 | x1.121 | x1.151 | 1.0 | h100_sxm | h100_sxm | x1.0 | good |

