## krauncher

11 tasks; versions: {"calibration_id": "c-d589e2f979e6", "client_commit": "0e2e207275f808b0e86fe9f0d49122ec993fe2d4"}

### Forecast time

median 0.62 s, p90 0.65 s, max 0.76 s

### Classification / assay fields

| field | match | differs in |
|---|---|---|
| workload_type | 11/11 | - |
| mode | 11/11 | - |
| framework | 11/11 | - |
| precision | 11/11 | - |
| params_billions | 11/11 | - |
| batch_size | 6/11 | krauncher_tutorials/phi3_inference, krauncher_tutorials/qwen15_inference, krauncher_tutorials/qwen25_7b_batched, krauncher_tutorials/qwen25_7b_gsm8k, krauncher_tutorials/qwen25_7b_long |
| epochs | 4/4 | - |
| dataset_samples | 7/8 | krauncher_tutorials/resnet152_food101 |
| seq_len | 3/3 | - |
| cpu_only | 11/11 | - |

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

136 (task, GPU) pairs: typical error x1.249, p90 x1.966; median rank correlation 0.83; pick slower than the fastest in 3/10 tasks

| task | GPUs | typical error | p90 | rank corr | picked | fastest | time regret | status |
|---|---|---|---|---|---|---|---|---|
| krauncher_tutorials/bert_batch_inference | 10 | x1.262 | x1.721 | 0.85 | rtx_6000_blackwell | rtx_6000_blackwell | x1.0 | good |
| krauncher_tutorials/bert_imdb | 27 | x1.246 | x1.65 | 0.93 | b200 | b200 | x1.0 | good |
| krauncher_tutorials/phi3_inference | 10 | x1.326 | x2.012 | 0.84 | rtx_6000_blackwell | rtx_6000_blackwell | x1.0 | good |
| krauncher_tutorials/qwen15_inference | 10 | x1.286 | x2.826 | 0.66 | h100_sxm | rtx_6000_blackwell | x2.031 | critical |
| krauncher_tutorials/qwen25_7b_batched | 9 | x1.179 | x1.818 | 0.87 | rtx_6000_blackwell | rtx_6000_blackwell | x1.0 | good |
| krauncher_tutorials/qwen25_7b_gsm8k | 12 | x1.153 | x1.705 | 0.83 | rtx_6000_blackwell | rtx_6000_blackwell | x1.0 | good |
| krauncher_tutorials/qwen25_7b_long | 8 | x1.099 | x1.671 | 0.55 | rtx_6000_blackwell | rtx_6000_blackwell | x1.0 | warning |
| krauncher_tutorials/qwen25_7b_lora_alpaca | 10 | x1.201 | x1.696 | 0.92 | b300 | b300 | x1.0 | good |
| krauncher_tutorials/resnet152_food101 | 28 | x1.421 | x2.246 | 0.57 | b200 | rtx_6000_blackwell | x1.653 | critical |
| krauncher_tutorials/vit_batch_inference | 12 | x1.406 | x2.47 | 0.66 | h100_sxm | rtx_6000_blackwell | x1.538 | critical |

#### Ladder coverage

measured GPUs without a forecast: krauncher_tutorials/qwen25_7b_lora_alpaca 2/12

### Measurements: lambda-dlb-v1-fp16 (independent: True; 1 tasks)

#### VRAM upper bound

forecast above the measured bound in 1/1 tasks; lambdalabs/resnet50_amp_bs1280: 130 GB > 80 GB

#### Ladder coverage

measured GPUs without a forecast: lambdalabs/resnet50_amp_bs1280 4/4

