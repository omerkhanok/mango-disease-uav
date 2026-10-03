# Hardware — Hexacopter UAV

## Overview

Custom hexacopter UAV designed and fabricated for mango leaf disease monitoring and targeted pesticide spraying.

## Specifications

| Component | Details |
|-----------|---------|
| Frame | Custom hexacopter, carbon fiber booms |
| Motors | 6x brushless (red/black housings) |
| Propellers | 2-blade, black |
| Flight Controller | Mounted on top plate |
| Camera | Integrated gimbal, white spherical housing |
| Spray Tank | Translucent white container |
| Pump | Relay-driven DC pump |
| Tubing | Pink flexible hose to nozzle |
| Landing Gear | Black legs with foam grips |
| Power | LiPo battery (top plate) |

## Why Hexacopter?

- Higher payload capacity than quadcopter
- Better flight stability
- Vertical take-off and landing (VTOL)
- Stable hover for image capture

## Flight Parameters

| Parameter | Value |
|-----------|-------|
| Altitude | 10-30 m above canopy |
| Image format | RGB, geo-tagged |
| Cropping | 224x224 |
| Normalization | ImageNet mean/std |
| Spray trigger | Relay -> pump -> nozzle |

## Photos

- `drone_side_view.jpg` — full drone with spray tank visible
- `drone_front_view.jpg` — front angle with camera gimbal

## Future Work

- Onboard NVIDIA Jetson Nano for real-time inference
- INT8 quantization + TensorRT
- Multi-instance detection (YOLO/DETR -> ViT)
- Swarm coordination for large orchards
