"""Model factory for 2D segmentation baselines (segmentation_models_pytorch)."""
import segmentation_models_pytorch as smp

# name -> (arch, encoder)
BASELINES = {
    "unet_r34":      ("Unet",          "resnet34"),
    "unetpp_r34":    ("UnetPlusPlus",  "resnet34"),
    "deeplabv3p_r34":("DeepLabV3Plus", "resnet34"),
    "segformer_b1":  ("Segformer",     "mit_b1"),
}

def build_model(name, in_channels, num_classes, encoder_weights="imagenet"):
    arch, encoder = BASELINES[name]
    return smp.create_model(
        arch=arch,
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
