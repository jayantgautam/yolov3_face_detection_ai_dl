
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm
import os


# test if pytorch computations can be done of the graphics card instead of the regular processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Class labels
class_labels = ["faces"]

# Defining a function to calculate Intersection over Union (IoU)
'''
This function calculates the Intersection over Union (IoU) score between two bounding boxes 
(predicted box and ground truth box). IoU is a common metric for evaluating object detection models, 
as it measures the overlap between two boxes.
'''
def iou(box1, box2):
    '''
    box1: The first bounding box (can be predicted or ground truth).
    
    box2: The second bounding box (either predicted or ground truth, depending on is_pred).
    
    is_pred=True: This flag indicates whether the IoU is being calculated between a prediction 
    (box1) and the ground truth (box2). If is_pred is True, then box1 is the prediction, 
    and box2 is the label (ground truth).
    '''

    # IoU score for prediction and label
    # box1 (prediction) and box2 (label) are both in [x, y, width, height] format
        
    # Box coordinates of prediction
    b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
    b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
    b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
    b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

    # Box coordinates of ground truth
    b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
    b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
    b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
    b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2
        
    '''
    b1_x1, b1_y1: These represent the top-left corner of the predicted box. 
    The x_center and y_center values are adjusted by subtracting half of the width and height 
    (width/2 and height/2).
        
    b1_x2, b1_y2: These represent the bottom-right corner of the predicted box. The x_center 
    and y_center values are adjusted by adding half of the width and height (width/2 and height/2).
    '''

    # Get the coordinates of the intersection rectangle
    x1 = torch.max(b1_x1, b2_x1)
    y1 = torch.max(b1_y1, b2_y1)
    x2 = torch.min(b1_x2, b2_x2)
    y2 = torch.min(b1_y2, b2_y2)
    # Make sure the intersection is at least 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        
    '''
    x1, y1: These are the coordinates of the top-left corner of the intersection. We take the maximum 
    of the two bounding boxes' top-left x and y coordinates to ensure we get the correct intersection.
    
    x2, y2: These are the coordinates of the bottom-right corner of the intersection. We take the 
    minimum of the two bounding boxes' bottom-right x and y coordinates to ensure the intersection 
    is bounded correctly.
        
    clamp(0): Ensures that the intersection dimensions are not negative. If x2 - x1 or y2 - y1 is 
    negative (indicating no overlap), it will be clamped to 0.

    intersection: The area of the intersection rectangle is calculated by multiplying the width and 
    height of the intersection.
    '''

    # Calculate the union area
    box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))
    union = box1_area + box2_area - intersection
        
    '''
    box1_area, box2_area: The areas of the individual bounding boxes (box1 and box2) are calculated 
    by multiplying their width and height.

    union: The union area is computed by adding the areas of box1 and box2 and subtracting the 
    intersection area (to avoid double-counting the overlap).
    '''

    # Calculate the IoU score
    epsilon = 1e-6
    iou_score = intersection / (union + epsilon)
        
    '''
    epsilon: A small value added to the denominator to avoid division by zero in case the union is 
    extremely small.

    iou_score: The IoU score is calculated by dividing the intersection area by the union area.
    '''

    # Return IoU score
    return iou_score


# Non-Maximum Suppression (NMS) function to remove overlapping bounding boxes
'''
This function is used to filter out overlapping bounding boxes that are considered redundant based 
on their IoU (Intersection over Union) and a confidence threshold. It helps to retain the best bounding 
boxes (typically with the highest confidence) while discarding the ones that overlap too much.
'''
def nms(bboxes, iou_threshold, threshold):
    bboxes = [box for box in bboxes if box[1] > threshold]
    
    '''
    This line performs a thresholding operation on a list of bounding boxes, each represented by 
    [class_pred, confidence, x, y, w, h] and filters out any bounding boxes where the confidence score 
    (box[1]) is below the given threshold. It keeps only the boxes that have a confidence score greater 
    than the threshold.
    '''
    
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    
    '''
    This line sorts the bounding boxes by their confidence score in descending order.
    '''
    
    # Initialize an empty list
    bboxes_nms = []

    while bboxes:
        # pops the first box from the sorted list bboxes, 
        # this is the box that we will "keep" after the NMS process
        chosen_box = bboxes.pop(0) 
        bboxes_nms.append(chosen_box)

        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0] or
            iou(torch.tensor(chosen_box[2:]), torch.tensor(box[2:])) < iou_threshold
        ]
        
    '''
    Iterate over the remaining bounding boxes and keeps only those that are not overlapping too much 
    (based on the IoU threshold). If the class of the current bounding box (box[0]) is different from 
    the chosen box's class (chosen_box[0]), we keep the box. If the class of the box is the same, we 
    compute the IoU between the chosen box and the current box. If the IoU is less than the iou_threshold, 
    we keep the box. This ensures that we discard boxes that overlap too much with the chosen box.
    '''    
    
    # The remaining boxes after NMS are returned
    return bboxes_nms


# Anchor boxes for each feature map scaled between 0 and 1
# 3 feature maps at 3 different scales based on YOLOv3 paper
'''
Anchor boxes are predefined bounding boxes with different aspect ratios and scales that are used to 
predict the location of objects in an image. During training, the model tries to match each object 
to one of the anchor boxes. The goal is to find which anchor box best fits the object in terms of 
size and shape. The idea is to use these predefined boxes as a starting point for the object 
detection predictions.
'''
ANCHORS = [
    [(0.56, 0.56), (0.622857, 0.622857), (0.685714, 0.685714)], # larger objects
    [(0.788571, 0.788571), (0.597143, 0.597143), (0.662857, 0.662857)], # medium-sized objects
    [(0.642857, 0.642857), (0.74, 0.74), (0.708571, 0.708571)], # small objects
]   

'''
3 feature maps at 3 different scales: This is because YOLOv3 uses multiple scales to detect objects 
of various sizes. The anchor boxes are assigned to three different feature map sizes (often corresponding 
to different layers in the network), allowing the model to detect objects from small to large.
'''

# Number of Epochs
epochs = 120

# Image Size
image_size = 160

# Grid cell sizes
s = [image_size // 32, image_size // 16, image_size // 8] 

'''
s calculates the spatial size of the three different feature maps at different scales, used for detecting 
objects of different sizes in the image.
'''

# Function to load dataset
'''
Inherits from torch.utils.data.Dataset, the base class for all PyTorch datasets.
Enables compatibility with PyTorch DataLoader for batching, shuffling, etc.
''' 
class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, anchors, image_size=160, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.anchors = anchors  # List of 3 lists (each for one scale)
        self.image_size = image_size
        self.images = os.listdir(image_dir)
        self.num_scales = 3  # YOLOv3 uses 3 prediction scales

    # Returns the number of images in the dataset (used by DataLoader)
    def __len__(self):
        return len(self.images)

    # Retrieves one sample (image + label) by index
    def __getitem__(self, idx):
        # Constructs full paths to the image and corresponding label file.
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].replace(".jpg", ".txt"))

        # Opens the image using PIL and ensures it's in RGB format
        image = Image.open(img_path).convert("RGB")
        boxes = []

        with open(label_path) as f:
            for line in f.readlines():
                class_label, x, y, w, h = map(float, line.strip().split())
                boxes.append([class_label, x, y, w, h])
        
        boxes = torch.tensor(boxes)
        
        '''
        Reads each label line (format: class x_center y_center width height — all normalized [0, 1]) 
        and stores them. 
        
        Converts to a PyTorch tensor of shape (num_boxes, 5).
        '''

        # Applies transformations (if any)
        if self.transform:
            image = self.transform(image)

        targets = self.encode_targets(boxes)
        return image, targets
        
        '''
        Converts boxes into target tensors for each scale using encode_targets() and returns them 
        with the image.
        '''
    
    # Converts bounding boxes into YOLO’s target format for 3 scales
    def encode_targets(self, boxes):
        """
        boxes: (N, 5) - [class_label, x, y, w, h] (normalized 0-1)
        Returns a tuple of 3 tensors (1 for each scale), each of shape: 
        (3, S, S, 6) - [objectness, x, y, w, h, class]
        """
        targets = []

        # Iterate over each prediction scale 
        for scale_idx, scale_anchors in enumerate(self.anchors):
            # Calculates the feature map size S for this scale (based on image size)
            # E.g., if image_size = 160, then S = 5, 10, 20 for different scales
            S = self.image_size // (2 ** (5 - scale_idx))
            
            num_anchors = len(scale_anchors)
            target = torch.zeros((num_anchors, S, S, 6))  # [p, x, y, w, h, class]

            '''
            Initializes the target tensor: shape (anchors_per_scale, S, S, 6).
            6 elements: [objectness, x, y, w, h, class_label].
            
            The target tensor in YOLO (specifically this implementation) has the shape: 
            (num_anchors, S, S, 6) And each of the 6 elements represents the information needed for 
            object detection at each grid cell and anchor box. Here's why it's 6 element:
                
            Index	Value	      Description
            0	    objectness	  1 if an object is present, 0 otherwise. YOLO learns to predict this.
            1	    x	          x-coordinate of the box center (normalized to [0, 1] in full image coords).
            2	    y	          y-coordinate of the box center.
            3	    w	          Width of the bounding box (normalized).
            4	    h	          Height of the bounding box (normalized).
            5	    class	      Class label index (as an integer, not one-hot encoded).    
            '''

            # For each box, find which grid cell it falls into at this scale
            for box in boxes:
                class_label, x, y, w, h = box.tolist()
                i = min(int(x * S), S - 1)
                j = min(int(y * S), S - 1)

                # Convert normalized width/height into pixel values
                box_w, box_h = w * self.image_size, h * self.image_size

                # For this box, find the anchor box with the highest IoU (intersection over union) 
                # among those for the current scale.
                best_iou = 0
                best_anchor_idx = 0
                for anchor_idx, (anchor_w, anchor_h) in enumerate(scale_anchors):
                    anchor_w_scaled = anchor_w * self.image_size
                    anchor_h_scaled = anchor_h * self.image_size
                    iou_score = self.iou(box_w, box_h, anchor_w_scaled, anchor_h_scaled)
                    if iou_score > best_iou:
                        best_iou = iou_score
                        best_anchor_idx = anchor_idx
                        
                        '''
                        TL;DR: YOLO chooses the anchor with the highest IoU with the ground-truth box 
                        because that anchor best matches the shape of the object — meaning it's the most 
                        appropriate one to be responsible for predicting it.
                        
                        YOLOv3 (and newer versions) predict boxes at each grid cell using predefined anchor 
                        boxes (prior shapes). These anchors differ in aspect ratios and sizes.

                        During training:
                        Each ground truth object must be assigned to exactly one anchor box on one scale.
                        This prevents multiple anchors or scales from predicting the same object 
                        (redundancy and confusion).
                        
                        The IoU (Intersection over Union) measures how similar two boxes are in size and 
                        aspect ratio. By assigning the object to the anchor with the highest IoU, we:

                        Ensure the best-shaped anchor is used.

                        Help the model converge faster and better — the anchor already resembles the 
                        ground truth, so the model needs smaller adjustments.

                        Avoid bad matches where an anchor is too wide, too tall, etc.
                        '''

                # Encode only one anchor per scale
                if target[best_anchor_idx, j, i, 0] == 0:
                    target[best_anchor_idx, j, i, 0] = 1
                    target[best_anchor_idx, j, i, 1:5] = torch.tensor([x, y, w, h])
                    target[best_anchor_idx, j, i, 5] = int(class_label)
                    
                    '''
                    If that cell-anchor combo hasn't been assigned yet, assign the object:
                    0: objectness score = 1, 1:5: box coordinates, 5: class label
                    '''
                    
            # Add the tensor for this scale to the targets list
            targets.append(target)

        # Returns a tuple of 3 tensors (one per scale)
        return tuple(targets)

    # Computes IoU between two boxes (assuming top-left aligned at origin)
    def iou(self, w1, h1, w2, h2):
        inter_w = min(w1, w2)
        inter_h = min(h1, h2)
        intersection = inter_w * inter_h

        # Calculates intersection and union
        union = w1 * h1 + w2 * h2 - intersection
        
        # Returns IoU score
        return intersection / union if union > 0 else 0


# Prepare data augmentation
train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

# Load the dataset and hand over directories, anchors, image size and transforms
train_dataset = YOLODataset(
    image_dir='data/faces/train/images',
    label_dir='data/faces/train/labels',
    anchors=ANCHORS,
    image_size=image_size,
    transform=train_transform
)

test_dataset = YOLODataset(
    image_dir='data/faces/val/images',
    label_dir='data/faces/val/labels',
    anchors=ANCHORS,
    image_size=image_size,
    transform=test_transform
)

# Define the batch size
batch_size = 16

# Load dataset to dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# CNN Block: Combines convolution, batch normalization, and activation.
'''
This is a PyTorch nn.Module subclass.
'''
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs):
        super().__init__()
        
        '''
        in_channels: input depth
        out_channels: output depth (number of filters)
        use_batch_norm: whether to include BatchNorm
        **kwargs: allows passing arguments like kernel_size, stride, padding, etc.
        '''

        # Applies a 2D convolution. 
        # If BatchNorm is used, bias=False because BatchNorm has its own bias/shift term.
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs)
        
        # Normalizes the output feature maps (if enabled)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Applies the Leaky ReLU activation function with a negative slope of 0.1
        self.activation = nn.LeakyReLU(0.1)
        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        # Applying convolution
        x = self.conv(x)
        # Applying BatchNorm and activation if needed
        if self.use_batch_norm:
            x = self.bn(x)
            return self.activation(x)
        else:
            return x


# Residual Block: Implements skip connections for better gradient flow and deeper networks
'''
A Residual Block is a building block of ResNet (Residual Network) that allows information to "skip over" 
layers via skip connections (also called shortcuts).

Instead of learning a direct mapping H(x), the block learns a residual function: F(x)=H(x)−x⇒H(x)=F(x)+x
This means the block learns what to add to the input x to get the output, not the output directly.
'''
class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        
        '''
        channels: Number of input/output channels (same, because residual adds inputs and outputs).
        use_residual: Enable/disable skip connections.
        num_repeats: Number of times to repeat the inner block (typically 1 or more).
        '''

        res_layers = []
        for _ in range(num_repeats):
            res_layers += [
                nn.Sequential(
                    nn.Conv2d(channels, channels // 2, kernel_size=1),
                    nn.BatchNorm2d(channels // 2),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU(0.1)
                )
            ]
        self.layers = nn.ModuleList(res_layers)
        self.use_residual = use_residual
        self.num_repeats = num_repeats
        
        '''
        channels: Number of input/output channels (same, because residual adds inputs and outputs).
        use_residual: Enable/disable skip connections.
        num_repeats: Number of times to repeat the inner block (typically 1 or more).
        '''
    # Defining forward pass for each residual block
    def forward(self, x):
        
        # Stores the input (residual)
        for layer in self.layers:
            residual = x
            
            # Passes x through the layer
            x = layer(x)
            
            # Adds the input back to the output (skip connection) if use_residual=True
            if self.use_residual:
                x = x + residual
        return x
  
    
# Defining scale prediction class
    '''
    This class is typically used in YOLO-style object detection networks, where predictions are made at 
    different scales for bounding boxes, class probabilities, and objectness. The ScalePrediction class is 
    a submodule that handles the prediction for a specific scale (or resolution) in the network. It outputs 
    a set of predictions in the format: batch_size, 3, grid_size, grid_size, num_classes + 5 (Please check 
    the comment below (Line: 277 - 290)) 
    '''
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        '''
        in_channels: Number of input channels (depth of the feature map from the previous layer).
        num_classes: Number of object classes that the model will predict.
        '''
        
        # Defining the layers in the network
        self.pred = nn.Sequential(
            nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*in_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(2*in_channels, (num_classes + 5) * 3, kernel_size=1),
        )
        
        '''
        First Conv2D Layer (nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1)):
        Applies a convolution with a kernel size of 3x3, with padding to preserve the spatial dimensions 
        (i.e., keeping the input and output sizes the same). It doubles the number of input channels.

        Batch Normalization (nn.BatchNorm2d): Normalizes the output from the previous convolution to 
        stabilize the learning process by reducing internal covariate shift.

        LeakyReLU Activation (nn.LeakyReLU(0.1)): Applies the LeakyReLU activation with a slope of 0.1 
        for negative values, allowing some negative values to pass through and avoiding dead neurons.

        Second Conv2D Layer (nn.Conv2d(2*in_channels, (num_classes + 5) * 3, kernel_size=1)):
        The second convolution reduces the depth to (num_classes + 5) * 3. The factor of 3 represents 
        the three anchor boxes (for YOLO, typically 3 anchors are used per grid cell).

        The reason for the num_classes + 5 is that: 5 corresponds to the 5 attributes predicted per 
        anchor box (objectness, x, y, w, h). num_classes corresponds to the predicted class probabilities.
        '''
        
        self.num_classes = num_classes
    
    # Defining the forward pass and reshaping the output to the desired output 
    # format: (batch_size, 3, grid_size, grid_size, num_classes + 5)
    def forward(self, x):
        
        # Passes the input tensor x through the layers defined in self.pred. The output shape will 
        # be (batch_size, (num_classes + 5) * 3, grid_height, grid_width)
        output = self.pred(x)
        
        # Reshapes the output to the desired format
        output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3))
        
        '''
        x.size(0) is the batch size.   
        The 3 represents the three anchor boxes per grid cell.
        self.num_classes + 5 represents the predicted attributes for each anchor 
        (5 for bounding box and objectness, plus num_classes for the class probability).
        x.size(2) and x.size(3) are the height and width of the feature map (grid size).
        
        After this reshaping, the output tensor will have the shape: batch_size, 3, grid_size, grid_size, num_classes + 5        
        '''
        
        # Reordering the num_classes + 5 dimension from the 3rd position to the last
        output = output.permute(0, 1, 3, 4, 2)
        
        return output    


# Class for defining YOLOv3 model, a simplified version of original YOLOv3 (Darknet-53) architecture
class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Layers list for YOLOv3
        self.layers = nn.ModuleList([
            CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1),
            CNNBlock(32, 64, kernel_size=3, stride=2, padding=1),
            
            # These progressively downsample the image (by stride=2) and increase feature richness
            ResidualBlock(64, num_repeats=1), 
            
            CNNBlock(64, 128, kernel_size=3, stride=2, padding=1),
            
            # These progressively downsample the image (by stride=2) and increase feature richness
            ResidualBlock(128, num_repeats=2),
            
            CNNBlock(128, 256, kernel_size=3, stride=2, padding=1),
            
            # These progressively downsample the image (by stride=2) and increase feature richness
            ResidualBlock(256, num_repeats=8),
            
            CNNBlock(256, 512, kernel_size=3, stride=2, padding=1),
            
            # These progressively downsample the image (by stride=2) and increase feature richness
            ResidualBlock(512, num_repeats=8),
            
            CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1),
            
            # These progressively downsample the image (by stride=2) and increase feature richness
            ResidualBlock(1024, num_repeats=4),
            
            # This is the first prediction head:
            # Outputs predictions at the lowest resolution (e.g., 160/32 = 5×5 for an input image size of 160×160).
            # Best for detecting large objects.    
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            ResidualBlock(1024, use_residual=False, num_repeats=1),                        
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
            ScalePrediction(512, num_classes=num_classes),

            # Upsample + Route Connection 1
            # Upsamples the previous feature map.
            # Concatenates with earlier layer (route_connections) for richer features.
            # Outputs predictions at a medium resolution (e.g., 160/16 = 10x10) — good for medium-sized objects.
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2),            
            CNNBlock(768, 256, kernel_size=1, stride=1, padding=0),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ResidualBlock(512, use_residual=False, num_repeats=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
            ScalePrediction(256, num_classes=num_classes),
            
            # Upsample + Route Connection 2
            # Upsample again.
            # Predicts at high resolution (e.g., 160/8 = 52×52) — good for small objects.
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2),
            CNNBlock(384, 128, kernel_size=1, stride=1, padding=0),
            CNNBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ResidualBlock(256, use_residual=False, num_repeats=1),
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0),
            ScalePrediction(128, num_classes=num_classes)
        ])
    
    # Forward pass for YOLOv3 with route connections and scale predictions
    def forward(self, x):
        outputs = []
        route_connections = []

        for layer in self.layers:
            
            # Prediction layers:
            # Collect predictions from the three ScalePrediction modules
            # Each one returns shape: (batch_size, 3, grid_size, grid_size, num_classes + 5)
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            
            # Normal layers: 
            # For CNNBlock or ResidualBlock (non-prediction layers), just apply the layer
            x = layer(x)

            # Skip (route) connections: 
            # Save outputs of ResidualBlocks with 8 repeats (these are 256 and 512-channel features from earlier)
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)
            
            # Upsample + concat: 
            # After upsampling, concatenate with saved earlier features
            # This is feature fusion, crucial for detecting small objects
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        return outputs
    
# Creating model and testing output shapes
model = YOLOv3(num_classes=1)
x = torch.randn((1, 3, image_size, image_size))
out = model(x)
print(out[0].shape)
print(out[1].shape)
print(out[2].shape)    


# Defining YOLO loss class
class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        
        '''
        This sets up the different components of the YOLO loss:
        MSELoss: For box coordinates and objectness confidence.
        BCEWithLogitsLoss: For binary classification (object or no object).
        CrossEntropyLoss: For multi-class classification (class prediction).
        Sigmoid: Used to squash outputs to the range [0, 1] where appropriate.
        '''

    def forward(self, pred, target, anchors):
        
        '''
        pred: Predictions from the model — shape (batch_size, 3, S, S, num_classes + 5)
        target: Ground truth labels in the same format.
        anchors: Anchor boxes for the current scale (3 anchors, each with width & height)
        '''

        # Identifying which cells in target have objects and which have no objects
        obj = target[..., 0] == 1  # Object mask
        no_obj = target[..., 0] == 0  # No-object mask

        '''
        Masking Object and No-object Cells:
        target[..., 0]: The objectness value (1 if object exists, 0 otherwise).
        These boolean masks are used to separate positive and negative samples.
        '''

        # No object loss
        no_object_loss = self.bce(pred[..., 0:1][no_obj], target[..., 0:1][no_obj])

        '''
        No-object loss (BCE):
        Penalizes the model for predicting high objectness where no object exists.
        Applied to all negative (background) cells.
        '''

        # Transform predictions
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        x = self.sigmoid(pred[..., 1])
        y = self.sigmoid(pred[..., 2])
        w = torch.exp(pred[..., 3]) * anchors[..., 0]
        h = torch.exp(pred[..., 4]) * anchors[..., 1]

        box_preds = torch.stack([x, y, w, h], dim=-1)

        '''
        Bounding Box Prediction:
        (x, y) predicted offsets relative to the top-left corner of each grid cell,
        passed through sigmoid to constrain them to (0, 1), meaning they stay within the cell.

        (w, h) predicted as log-space deviations from the anchor box.
        Applying `exp` and multiplying by the anchor dimensions recovers the final width and height.

        Result: box_preds = [x, y, w, h] format, normalized to the feature map size,
        matching the format used by the targets.
        '''
        
        # Calculate IoU (intersection over union) for objectness score
        ious = iou(box_preds[obj], target[..., 1:5][obj]).detach()
        
        '''
        IoU for Object Cells:
        Computes IoU between predicted boxes and ground truth boxes for positive cells only.
        detach() so it's not part of the gradient (IoU is used as a target, not an input).
        '''

        # Calculating Object loss
        object_loss = self.mse(self.sigmoid(pred[..., 0][obj]), ious.squeeze(-1))

        '''
        Objectness Loss (MSE between predicted confidence and IoU):
        Predicts objectness confidence, i.e., how confident the model is that an object exists.
        It's trained to match the IoU score (not just 1), as this helps learn better localization confidence.
        '''

        # Box loss
        target_box = target[..., 1:5].clone()
        target_box[..., 2:] = torch.log((target_box[..., 2:] + 1e-6) / anchors[0])
        pred_box = pred[..., 1:5].clone()
        pred_box[..., :2] = self.sigmoid(pred_box[..., :2])

        box_loss = self.mse(pred_box[obj], target_box[obj])
        
        '''
        Coordinate Loss (Box regression):
        Center coords (x, y) are passed through sigmoid.
        Width/height (w, h) are converted to the log scale, as the model learns these in log space.
        MSE is applied to [x, y, w, h] predictions only where objects are present.
        '''

        # Class loss — only 1 class, so use BCE
        class_loss = self.bce(pred[..., 5][obj], target[..., 5][obj])

        '''
        Class Prediction Loss:
        Applied only on cells that contain objects.
        Targets should contain class indices, and this predicts the correct class for each object.
        '''

        return box_loss + object_loss + no_object_loss + class_loss
    
        '''
        In a full YOLOv3 implementation, these losses might be weighted differently 
        (e.g., λ_obj, λ_noobj, etc.), but here they are added equally.
        '''

      
# Define the train function to train the model
'''
Train the model with mixed-precision training to speed things up and save memory.
'''
def training_loop(loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    
    '''
    loader: DataLoader with training data.
    model: The YOLOv3 model.
    optimizer: For updating weights.
    loss_fn: Custom YOLO loss function.
    scaler: For mixed precision training.
    scaled_anchors: Anchors scaled to different feature map sizes.
    '''
    
    # Creates a progress bar (from tqdm) to visualize training progress.
    progress_bar = tqdm(loader, leave=True)

    # Initializing a list to store the losses
    losses = []

    # Training Loop Over Data:
    # Iterates over the DataLoader, returning input images x and targets y
    for _, (x, y) in enumerate(progress_bar):
        
        # Moves images to GPU (or CPU, depending on device)
        x = x.to(device)
        y0, y1, y2 = (y[0].to(device), y[1].to(device), y[2].to(device))

        '''
        YOLOv3 makes predictions at 3 scales. This line splits target labels for each scale and moves 
        them to the same device.
        '''

        # Forward Pass (Mixed Precision):
        # Enables automatic mixed precision for the block below — speeds up training and reduces memory
        with torch.cuda.amp.autocast():
            
            # Forward pass: gets predictions from the model for input x
            outputs = model(x)
            
            # Computes total loss by summing losses at each scale (3 scales in YOLOv3)
            loss = (
                  loss_fn(outputs[0], y0, scaled_anchors[0])
                + loss_fn(outputs[1], y1, scaled_anchors[1])
                + loss_fn(outputs[2], y2, scaled_anchors[2])
            )

        # Backward Pass and Optimization:
        # Stores the current loss (as a number) in the list
        losses.append(loss.item())

        # Resets gradients from the previous iteration
        optimizer.zero_grad()

        # Scales the loss (for mixed precision) and performs backpropagation to compute gradients
        scaler.scale(loss).backward()

        # Scales and applies the optimizer step to update weights
        scaler.step(optimizer)

        # Updates the scaler for the next iteration (important for numerical stability)
        scaler.update()

        # Update Progress Bar:
        # Calculates average loss so far and displays it in the progress bar
        mean_loss = sum(losses) / len(losses)
        progress_bar.set_postfix(loss=mean_loss)    
        

# Model, Optimizer, Loss Function Setup:        
# Creates the YOLOv3 model and moves it to GPU/CPU
model = YOLOv3().to(device)

# Uses Adam optimizer with a small learning rate
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)

# Initializes the YOLO-specific loss function
loss_fn = YOLOLoss()

# Prepares the scaler for mixed precision training
scaler = torch.cuda.amp.GradScaler()

# DataLoader Setup:
# Defining the train data loader
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = batch_size,
    num_workers = 0,
    shuffle = True,
    pin_memory = True,
)

'''
batch_size: How many samples per batch.
shuffle=True: Shuffle dataset every epoch.
pin_memory=True: Speed up host to device transfer.
'''

# Scaling the anchors
scaled_anchors = (torch.tensor(ANCHORS) * torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1,3,2)).to(device)

'''
Scales predefined anchors (ANCHORS) according to the output feature map sizes s for the 3 YOLO scales. 
This is needed because each YOLO head outputs predictions for a different resolution.
'''
# Load checkpoint if exists 
if os.path.exists("checkpoint.pth"):
    checkpoint = torch.load("checkpoint.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming from epoch {start_epoch}")
        
# Train the model for 20 epochs by calling training_loop repeatedly
# for epoch in range(1, epochs+1):
    
#     print("Epoch:", epoch)
#     training_loop(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
    
#     # Save checkpoint
#     torch.save({
#     'epoch': epoch,
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'scaler_state_dict': scaler.state_dict(),
# }, 'checkpoint.pth')


# Evaluate the model
# Helper to convert predictions from YOLO format to bounding boxes
def convert_predictions(pred, anchors, S, threshold):
    """
    Convert model predictions into bounding boxes.
    Args:
        pred: Tensor of shape (1, 3, S, S, 6)
        anchors: Tensor of shape (3, 2)
        S: grid size
        threshold: confidence threshold
    Returns:
        list of boxes: [class, confidence, x, y, w, h]
    """
    bboxes = []

    pred = pred.squeeze(0).cpu()  # (3, S, S, 6)
    anchors = anchors.cpu()

    for anchor_idx in range(3):
        for i in range(S):
            for j in range(S):
                objectness = torch.sigmoid(pred[anchor_idx, i, j, 0])

                if objectness < threshold:
                    continue

                x = (torch.sigmoid(pred[anchor_idx, i, j, 1]) + j) / S
                y = (torch.sigmoid(pred[anchor_idx, i, j, 2]) + i) / S

                w = anchors[anchor_idx][0] * torch.exp(torch.clamp(pred[anchor_idx, i, j, 3], max=4))
                h = anchors[anchor_idx][1] * torch.exp(torch.clamp(pred[anchor_idx, i, j, 4], max=4))

                # If only one class, class index is always 0
                class_pred = 0
                bboxes.append([class_pred, objectness.item(), x.item(), y.item(), w.item(), h.item()])

    # print(f"Box: conf={objectness:.2f}, x={x:.2f}, y={y:.2f}, w={w:.2f}, h={h:.2f}")

    return bboxes


# This function draws the predicted and ground truth boxes on the image
def visualize_image_with_boxes(image_tensor, pred_boxes, target_boxes):
    
    # Converts the image tensor to a PIL image
    image = TF.to_pil_image(image_tensor)
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    ax = plt.gca()

    # Draw predicted boxes in red
    for box in pred_boxes:
        x, y, w, h = box[2:]
        x0 = (x - w / 2) * image_size
        y0 = (y - h / 2) * image_size
        rect = patches.Rectangle((x0, y0), w * image_size, h * image_size, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    # Draw ground truth boxes in green
    for box in target_boxes:
        x, y, w, h = box
        x0 = (x - w / 2) * image_size
        y0 = (y - h / 2) * image_size
        rect = patches.Rectangle((x0, y0), w * image_size, h * image_size, linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)

    plt.show()


# Function to evaluate the model over a test dataset
def test_loop(model, loader, iou_threshold, conf_threshold, anchors, plot=False):
    
    # Puts model in evaluation mode
    model.eval()
    
    # Prepares a list to store IoUs
    mean_ious = []
    
    # Disables gradient tracking for faster evaluation
    with torch.no_grad():
        
        # Main Evaluation Loop:
        # Iterates through test dataset
        for images, targets in tqdm(loader, desc="Testing"):
            
            # Moves inputs to device and gets predictions
            images = images.to(device)
            batch_preds = model(images)

            # Decode predictions for each image
            for b in range(images.size(0)):
                boxes = []
                
                # Processes predictions from all 3 YOLO scales and collects bounding boxes
                for scale_idx in range(3):
                    S = images.shape[2] // (2 ** (5 - scale_idx))  # 160 / 32, 16, 8
                    anchor_set = anchors[scale_idx]
                    preds_scale = batch_preds[scale_idx][b].unsqueeze(0)
                    boxes += convert_predictions(preds_scale, anchor_set, S, conf_threshold)

                # Applies NMS to filter overlapping boxes based on IoU and confidence
                boxes = nms(boxes, iou_threshold, conf_threshold)

                # Extract Ground Truth Boxes, converts YOLO target format to box coordinates
                target_boxes = []
                for scale_target in targets:
                    target = scale_target[b]
                    obj_indices = (target[..., 0] == 1).nonzero(as_tuple=False)
                    for anchor_idx, y, x in obj_indices:
                        x_center = target[anchor_idx, y, x, 1].item()
                        y_center = target[anchor_idx, y, x, 2].item()
                        w = target[anchor_idx, y, x, 3].item()
                        h = target[anchor_idx, y, x, 4].item()
                        target_boxes.append(torch.tensor([x_center, y_center, w, h]))

                if len(boxes) == 0 or len(target_boxes) == 0:
                    continue

                # Compute IoU for Each Prediction, compares each predicted box with all ground truth boxes,                
                # recording the best IoU
                ious = []
                for pred_box in boxes:
                    pred_tensor = torch.tensor(pred_box[2:])
                    best_iou = 0
                    for gt_box in target_boxes:
                        score = iou(pred_tensor.unsqueeze(0), gt_box.unsqueeze(0))
                        best_iou = max(best_iou, score.item())
                    ious.append(best_iou)

                # Computes average IoU per image and adds to list
                if ious:
                    mean_ious.append(sum(ious) / len(ious))

                # Draws images with predicted and ground truth boxes if plot is true
                if plot:
                    visualize_image_with_boxes(images[b].cpu(), boxes, target_boxes)

    # Prints the average IoU and switches model back to training mode
    print(f"Mean IoU across test set: {sum(mean_ious) / len(mean_ious):.4f}")
    model.train()

   
# Run Evaluation:
# Sets up the test data loader (batch size = 1 for evaluation)   
""" 
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

# Calls the test loop to evaluate the model and optionally visualize predictions
test_loop(model, test_loader,
          iou_threshold=0.6,  # try 0.6 or 0.7 to be more aggressive
          conf_threshold=0.6,  # try 0.5 or even 0.6 for stricter filtering
          anchors=scaled_anchors,
          plot=True)
          """



# EXPORT ONNX MODEL
# This wrapper doesn't change your base model or training pipeline — it's just for ONNX export
class YOLOv3ExportWrapper(nn.Module):
    def __init__(self, yolov3_model):
        super().__init__()
        self.model = yolov3_model

    def forward(self, x):
        outputs = self.model(x)  # List of 3 outputs

        # Flatten each: (B, 3, S, S, 6) → (B, -1, 6)
        outputs_flattened = [o.reshape(o.size(0), -1, o.size(-1)) for o in outputs]

        # Concatenate: (B, N1 + N2 + N3, 6)
        out = torch.cat(outputs_flattened, dim=1)

        return out
    

# Use this wrapper during export only
# Load the full checkpoint (a dictionary)
checkpoint = torch.load('checkpoint.pth', map_location=device)

# Extract and load only the model weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

export_model = YOLOv3ExportWrapper(model).to(device)

dummy_input = torch.randn(1, 3, 160, 160).to(device)

torch.onnx.export(
    export_model,
    dummy_input,
    "YOLO_v3_face.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    dynamic_axes=None

)    
