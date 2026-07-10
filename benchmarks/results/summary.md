# Muonet benchmark results

## Dosage curve (tungsten_cube, realistic_sipm detector)

| muons | method | PSNR (dB) | SSIM | IoU | high-Z acc |
|------:|:-------|----------:|-----:|----:|-----------:|
| 128 | poca | 14.26 | 0.632 | 0.031 | 0.959 |
| 128 | learned | 12.81 | 0.020 | 0.349 | 0.963 |
| 256 | poca | 14.87 | 0.511 | 0.084 | 0.965 |
| 256 | learned | 14.83 | 0.028 | 0.423 | 0.973 |
| 512 | poca | 15.02 | 0.396 | 0.097 | 0.963 |
| 512 | learned | 14.99 | 0.031 | 0.532 | 0.978 |
| 1024 | poca | 15.09 | 0.247 | 0.130 | 0.964 |
| 1024 | learned | 15.27 | 0.033 | 0.542 | 0.975 |
| 4096 | poca | 16.21 | 0.216 | 0.210 | 0.969 |
| 4096 | learned | 15.69 | 0.035 | 0.513 | 0.973 |
| 8192 | poca | 16.46 | 0.198 | 0.262 | 0.971 |
| 8192 | learned | 16.08 | 0.035 | 0.492 | 0.973 |

## Realism ablation (tungsten_cube, 1024 muons)

| detector | method | PSNR (dB) | SSIM | IoU |
|:---------|:-------|----------:|-----:|----:|
| ideal | poca | 15.82 | 0.845 | 0.160 |
| ideal | learned | 14.86 | 0.030 | 0.390 |
| research | poca | 15.61 | 0.401 | 0.171 |
| research | learned | 15.49 | 0.033 | 0.509 |
| realistic_sipm | poca | 15.09 | 0.247 | 0.130 |
| realistic_sipm | learned | 15.27 | 0.033 | 0.542 |
| battery_limited | poca | 14.05 | 0.197 | 0.033 |
| battery_limited | learned | 12.49 | 0.013 | 0.035 |
