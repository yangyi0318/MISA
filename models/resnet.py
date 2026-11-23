import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class BasicBlock(nn.Module):
    """
    A standard basic block for ResNet, consisting of two 3x3 convolutions.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # Shortcut connection to match dimensions if needed
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    The main ResNet model class.
    **MODIFIED** to be adaptable for different input sizes.
    """

    def __init__(self, block, num_blocks, num_classes=10, input_size=32):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.original_input_size = input_size

        # --- MODIFICATION: Conditional Architecture based on input_size ---
        # If the input size is large (like ImageNet's 224x224), use the standard ResNet stem.
        # Otherwise, use the modified stem suitable for smaller images (like CIFAR-10's 32x32).
        if input_size > 64:  # Standard ResNet for ImageNet-sized images
            # Use a 7x7 conv with stride 2, followed by max pooling
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # Downsampling is done by: conv1 (stride 2), maxpool (stride 2),
            # layer2 (stride 2), layer3 (stride 2), and layer4 (stride 2).
            # Total downsampling factor = 2 * 2 * 2 * 2 * 2 = 32
            self.downsampling_factor = 32
        else:  # Modified ResNet for small images like CIFAR-10
            # Use a 3x3 conv with stride 1 and no max pooling
            # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()  # A layer that does nothing, acts as a placeholder
            # Downsampling is done by: layer2 (stride 2), layer3 (stride 2), layer4 (stride 2).
            # Total downsampling factor = 2 * 2 * 2 = 8
            self.downsampling_factor = 8

        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_val in strides:
            layers.append(block(self.in_planes, planes, stride_val))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_feature_map(self, x):
        """
        Runs the model up to the final convolutional layer to get the feature map.
        **MODIFIED** to include the conditional max pooling layer.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out) # This will be nn.Identity() for small inputs
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    def forward(self, x):
        """
        Standard forward pass for training and classification.
        """
        feature_map = self.get_feature_map(x)
        out = F.adaptive_avg_pool2d(feature_map, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def extract_features(self, x, region=None):
        """
        **MODIFIED METHOD**
        Extracts features from a region specified in ORIGINAL IMAGE coordinates.
        This method now works correctly with the dynamic downsampling factor.
        """
        feature_map = self.get_feature_map(x)

        if region is not None:
            if not (isinstance(region, (list, tuple)) and len(region) == 4):
                raise ValueError("Region must be a tuple or list of four integers (x1, y1, x2, y2).")

            x1_img, y1_img, x2_img, y2_img = region

            # Validate image-space coordinates
            if not (0 <= x1_img <= x2_img and x2_img < self.original_input_size and \
                    0 <= y1_img <= y2_img and y2_img < self.original_input_size):
                warnings.warn(
                    f"Image-space region {region} is invalid or out of bounds for input size "
                    f"{self.original_input_size}. Returning global features."
                )
                features = F.adaptive_avg_pool2d(feature_map, (1, 1))
                return features.view(features.size(0), -1)

            # Convert image-space coordinates to feature-map-space coordinates
            x_map_start = x1_img // self.downsampling_factor
            y_map_start = y1_img // self.downsampling_factor
            x_map_end = x2_img // self.downsampling_factor
            y_map_end = y2_img // self.downsampling_factor

            # Calculate width and height on the feature map, ensuring it's at least 1x1
            w_map = max(1, x_map_end - x_map_start + 1)
            h_map = max(1, y_map_end - y_map_start + 1)

            # Slice the feature map
            regional_feature_map = feature_map[:, :, y_map_start:y_map_start + h_map, x_map_start:x_map_start + w_map]

            # Pool the selected region to get a single feature vector
            features = F.adaptive_avg_pool2d(regional_feature_map, (1, 1))
            features = features.view(features.size(0), -1)

        else:
            # Default case: global features
            features = F.adaptive_avg_pool2d(feature_map, (1, 1))
            features = features.view(features.size(0), -1)

        return features


def resnet18(num_classes, input_size=32):
    """
    Instantiates a ResNet-18 model.
    The model will adapt based on the provided input_size.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, input_size=input_size)

# --- Example Usage ---
if __name__ == '__main__':
    # 1. Create a model for 32x32 images (like CIFAR-10)
    print("--- Model for 32x32 input ---")
    model_32 = resnet18(num_classes=10, input_size=32)
    print(model_32)
    print(f"Downsampling Factor: {model_32.downsampling_factor}")
    dummy_input_32 = torch.randn(1, 3, 32, 32)
    output_32 = model_32(dummy_input_32)
    print(f"Output shape for 32x32 input: {output_32.shape}\n")

    # Extract features from a region on the 32x32 image
    # Region is (x1, y1, x2, y2) = (8, 8, 23, 23)
    region_features_32 = model_32.extract_features(dummy_input_32, region=(8, 8, 23, 23))
    print(f"Regional feature shape for 32x32 input: {region_features_32.shape}\n")


    # 2. Create a model for 224x224 images (like ImageNet)
    print("--- Model for 224x224 input ---")
    model_224 = resnet18(num_classes=1000, input_size=224)
    # Note: The printout won't show the nn.Identity() for the 32x32 case,
    # but it will show the MaxPool2d for the 224x224 case.
    print(model_224)
    print(f"Downsampling Factor: {model_224.downsampling_factor}")
    dummy_input_224 = torch.randn(1, 3, 224, 224)
    output_224 = model_224(dummy_input_224)
    print(f"Output shape for 224x224 input: {output_224.shape}\n")

    # Extract features from a region on the 224x224 image
    # Region is (x1, y1, x2, y2) = (50, 50, 150, 150)
    region_features_224 = model_224.extract_features(dummy_input_224, region=(50, 50, 150, 150))
    print(f"Regional feature shape for 224x224 input: {region_features_224.shape}\n")