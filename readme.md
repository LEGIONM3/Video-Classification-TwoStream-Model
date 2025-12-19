# Two-Stream Violence Detection Network

## Model Architecture
- **Type**: Two-Stream Network (Spatial + Temporal)
- **Streams**:
  1. **RGB Stream**: ResNet3D (r3d_18) to process raw video frames. Captures appearance info.
  2. **Optical Flow Stream**: ResNet3D (r3d_18) to process computed dense optical flow. Captures motion info.
- **Fusion**: Features from both streams are concatenated and passed through fully connected layers.
- **Input**: 16 Frames (RGB) + 16 Flow Fields (Computed on the fly).
- **Computation**: Optical flow is computed using Farneback algorithm within the Dataloader.

## Dataset Structure
Expects `Dataset` folder in parent directory.
```
Dataset/
├── violence/
└── no-violence/
```

## How to Run
1. Install dependencies: `torch`, `opencv-python` (with contrib if needed for some algorithms, but Farneback is standard), `torchvision`.
2. Run `python train.py`.

##HuggingFace
1.Link: https://huggingface.co/LEGIONM36/Video-Classification-Two-Stream-Model

