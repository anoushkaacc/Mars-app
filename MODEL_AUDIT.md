# Model and Checkpoint Compatibility Audit

## Supported checkpoints

| Checkpoint | Architecture | Classes | State tensors |
|---|---|---:|---:|
| unet.pth | UNet5 | 5 | 118 |
| unet_cbam.pth | CBAMUNet5 | 5 | 163 |
| CCHNet_FP16.pth | YNet | 12 | 549 |
| binary_y_net_mars.pth | BinaryYNet | 12 | 549 |
| adaptive_y_net_mars.pth | AdaptiveQuantYNet | 12 | 562 |

## Structural verification

The model implementations in `models.py` were compared against the tensor names and shapes recorded in the supplied checkpoint inspection report.

All five supported architectures matched the reported tensor inventories:
- Missing keys: 0
- Unexpected keys: 0
- Shape mismatches: 0

Forward-pass smoke tests were also performed on the five model architectures.

## Important limitations

1. The actual checkpoint binaries were not available in this environment. Structural compatibility was checked against the inspection report, not by loading the user's local checkpoint values.
2. `y_net_mars.pth` is excluded. Its architecture could not be established reliably from the supplied report.
3. The five-class semantic mapping is based on the supplied notebooks. The original training-time class order for `unet.pth` was not independently established.
4. The twelve-class models use generic `Class 0` through `Class 11` labels because semantic names were not established in the supplied training scripts.
5. Test images are unlabelled. The application can display predictions and visual comparisons but cannot calculate ground-truth mIoU, Dice, or accuracy.

## Loading behavior

The application uses strict checkpoint loading. A mismatch in the actual checkpoint will raise an error rather than silently loading an incompatible architecture.