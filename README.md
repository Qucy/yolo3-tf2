## YoloV3 - an object detection algorithm implemented via TF 2.x

###### [source code](https://github.com/Qucy/yolo3-tf2)

In this article I assume you've already familiar with basic computer vision knowledge like convolutional layers, pooling layers, residual blocks, Yolov1 and YoloV2 (Yolo 9000).



### 1. YoloV3 overall architectural

Before we start to understand the details of YoloV3 we need to have an overall picture of YoloV3 and how does it works. Below picture depicting the over architecture of YoloV3. It can be divide into 3 parts, Darknet53, FPN (Feature Pyramid Network) and Yolo heads.

![yolov3_architecture_1](https://github.com/Qucy/yolo3-tf2/blob/master/img/yolov3_architecture_1.jpg)

- Darknet53 is also called backbone network, mainly is used for extracting features from source image, in YoloV3 it will generate 3 different feature maps which are (13, 13, 1024), (26, 26, 512) and (52, 52, 256).
- FPN is also called neck network, mainly used for concat and transform features from backbone network, feature maps will do up sampling and concat with features from other layers to enhance features.
- Yolo head is also call head network, used for classification and regression. From Darknet53 and FPN, model can generated 3 enhanced feature maps and their shape are (52, 52, 128), (26, 26, 256) and (13, 13, 512). Every feature map has 3 dimension, width, height and channels. Yolo head will make prediction on these feature maps, to predict whether there is an object in the grid, their category and bounding box coordinates.



### 2. YoloV3 architectural analyze

#### 2.1 Darknet53 - Backbone network

YoloV3 using Darknet53 as it's backbone network and in overall it has 2 main important features:

- residual block: Darknet53's residual block can be divided into 2 parts, the main inputs(x) will first do 1x1 convolution and 3x3 convolution, the residual parts will do nothing and concat with outputs(y) directly. In above image x1 , x2 and x8 means how many residual blocks we're going to repeat.

  ```python
  #---------------------------------------------------------------------#
  #   Residual Block
  #   Use ZeroPadding2D and a Conv with strides of 2 to reduce image height and width
  #   Loop num_blocks to produce multiple residual blocks
  #---------------------------------------------------------------------#
  def resblock_body(x, num_filters, num_blocks):
      x = ZeroPadding2D(((1,0),(1,0)))(x)
      x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
      for i in range(num_blocks):
          y = DarknetConv2D_BN_Leaky(num_filters//2, (1,1))(x) # 1x1 conv
          y = DarknetConv2D_BN_Leaky(num_filters, (3,3))(y) # 3x3 conv
          x = Add()([x,y]) # concat shortcut with main outputs
      return x
  ```

  ![residual_block](https://github.com/Qucy/yolo3-tf2/blob/master/img/residual_block.png)



- batch normalization and LeakyReLU: every DarknetConv2D block will followed by a Batch Normalization layer and LeakyReLU layer. The difference between ReLU and LeakyReLU is, LeakyReLU will still have non-zero slope when x is negative.

  ![LeakyReLU](https://github.com/Qucy/yolo3-tf2/blob/master/img/LeakyReLU.png)

DarknetConv2D's source code is as below

```python
#---------------------------------------------------#
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( # Compose is same as Sequential in keras
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))
```

The whole backbone network source code is as below

```python
from functools import wraps

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Add, BatchNormalization, Conv2D, LeakyReLU, ZeroPadding2D)
from tensorflow.keras.regularizers import l2
from utils.utils import compose

#------------------------------------------------------#
#   Single DarknetConv2D Block
#   DarknetConv2D
#------------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : RandomNormal(stddev=0.02), 'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

#---------------------------------------------------------------------#
#   Residual Block
#   Use ZeroPadding2D and a Conv with strides of 2 to reduce image height and width
#   Loop num_blocks to produce multiple residual blocks
#---------------------------------------------------------------------#
def resblock_body(x, num_filters, num_blocks):
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = DarknetConv2D_BN_Leaky(num_filters//2, (1,1))(x)
        y = DarknetConv2D_BN_Leaky(num_filters, (3,3))(y)
        x = Add()([x,y])
    return x

#---------------------------------------------------#
#   darknet53 backbone network
#   inputs image size = 416x416x3
#   output 3 feature maps
#---------------------------------------------------#
def darknet_body(x):
    # 416,416,3 -> 416,416,32
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    # 416,416,32 -> 208,208,64
    x = resblock_body(x, 64, 1)
    # 208,208,64 -> 104,104,128
    x = resblock_body(x, 128, 2)
    # 104,104,128 -> 52,52,256
    x = resblock_body(x, 256, 8)
    feat1 = x
    # 52,52,256 -> 26,26,512
    x = resblock_body(x, 512, 8)
    feat2 = x
    # 26,26,512 -> 13,13,1024
    x = resblock_body(x, 1024, 4)
    feat3 = x
    return feat1, feat2, feat3
```



#### 2.2 FPN - Extract and enhance feature maps at different scale

![yolov3_architecture_1](https://github.com/Qucy/yolo3-tf2/blob/master/img/yolov3_architecture_1.jpg)

In the FPN network, YoloV3 extract 3 different feature maps from backbone network. The reason behind this is that different shape help model to detect different size of bounding box.

- Middle layer feature, shape is (52, 52, 256), help model to detect small size bounding box.
- Lower layer feature, shape is (26, 26, 512), help model to detect middle size bounding box.
- bottom layer feature, shape is (13, 13, 1024), help mode to detect large size bounding box.

After retrieved 3 different feature maps from backbone network:

- First use bottom layer feature (13, 13, 1024) do convolution for 5 times and pass outputs to Yolo head to make prediction. The outputs will also pass to up sampling layers and then concat with lower layer feature (26, 26, 512) to produce a merged feature (26, 26, 768).
- Merged feature (26, 26, 768) do convolution for 5 times and pass outputs to Yolo head to make prediction. The outputs will also pass to up sampling layers and then concat with middle layer feature (13, 13, 256) to produce a merged feature (13, 13, 384).
- Merged feature (13, 13, 384) do convolution for 5 times and pass outputs to Yolo head to make prediction.



#### 2.3 Yolo head - generate predictions

By using FPN we can generated three enhanced feature maps with different shape, their shape are (13, 13, 255), (26, 26, 255) and (52, 52, 255), and Yolo head will use these feature maps to produce final predictions.

Yolo head will basically do 2 convolution operations, the first convolution layer using 3x3 filters and mainly for feature ensemble and second convolution layer using 1x1 filters mainly for adjust channels to match the prediction result.

If we are using VOC dataset, the final out from Yolo head should be (13, 13, 75), (26, 26, 75) and (52, 52, 75). The last dimension 75 is decided by the dataset, because VOC dataset has 20 different classes and each grid has 3 anchors, so the final dimension will be 3 * (20 + 1 + 4) =75.

If we are using COCO dataset, the final dimension will be 255, because COCO datasets has 80 classes, 3 * (80 + 1 + 4) = 255.

So to wrap up(assuming we're using COCO dataset), after input N 416*416 images into model, model will output 3 predictions with shape (N, 13, 13, 255), (N, 26, 26, 255) and (N, 52, 52, 255), each 255 will map to 3 anchors in the 13x13, 26x26 and 52x52 grids.

Below is the source code for FPN and Yolo head

```python
from tensorflow.keras.layers import Concatenate, Input, Lambda, UpSampling2D
from tensorflow.keras.models import Model
from utils.utils import compose

from nets.darknet import DarknetConv2D, DarknetConv2D_BN_Leaky, darknet_body
from nets.yolo_training import yolo_loss


#---------------------------------------------------#
#   Conv * 5
#---------------------------------------------------#
def make_five_conv(x, num_filters):
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    return x

#---------------------------------------------------#
#   Generate Yolo head
#---------------------------------------------------#
def make_yolo_head(x, num_filters, out_filters):
    y = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    # (80+1+4)*3=255 or (20+1+4)*3=85
    y = DarknetConv2D(out_filters, (1,1))(y)
    return y

#---------------------------------------------------#
#   Construct FPN network and prediction result
#---------------------------------------------------#
def yolo_body(input_shape, anchors_mask, num_classes):
    inputs      = Input(input_shape)
    #---------------------------------------------------#
    #   retrieve 3 feature maps from backbone network
    #   shape are：
    #   C3 => 52,52,256
    #   C4 => 26,26,512
    #   C5 => 13,13,1024
    #---------------------------------------------------#
    C3, C4, C5  = darknet_body(inputs)

    #---------------------------------------------------#
    #   Generate first FPN feature => P5 => (batch_size,13,13,3,85)
    #---------------------------------------------------#
    # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    x   = make_five_conv(C5, 512)
    P5  = make_yolo_head(x, 512, len(anchors_mask[0]) * (num_classes+5))

    # 13,13,512 -> 13,13,256 -> 26,26,256
    x   = compose(DarknetConv2D_BN_Leaky(256, (1,1)), UpSampling2D(2))(x)

    # 26,26,256 + 26,26,512 -> 26,26,768
    x   = Concatenate()([x, C4])
    #---------------------------------------------------#
    #   Generate second FPN feature => P4 => (batch_size,26,26,3,85)
    #---------------------------------------------------#
    # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    x   = make_five_conv(x, 256)
    P4  = make_yolo_head(x, 256, len(anchors_mask[1]) * (num_classes+5))

    # 26,26,256 -> 26,26,128 -> 52,52,128
    x   = compose(DarknetConv2D_BN_Leaky(128, (1,1)), UpSampling2D(2))(x)
    # 52,52,128 + 52,52,256 -> 52,52,384
    x   = Concatenate()([x, C3])
    #---------------------------------------------------#
    #   Generate second FPN feature => P3 => (batch_size,52,52,3,85)
    #---------------------------------------------------#
    # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
    x   = make_five_conv(x, 128)
    P3  = make_yolo_head(x, 128, len(anchors_mask[2]) * (num_classes+5))
    return Model(inputs, [P5, P4, P3])
```



### 3. Decoding Yolo V3 prediction result

After model generate prediction result, can we use it directly ？No, we still need to do some transformation or decode our prediction result before we can use it. Before decode prediction result, let's take a look at anchor boxes in YoloV3.

#### 3.1 Anchor boxes

From YoloV3 network it can generate 3 prediction with below shapes if we using COCO datasets:

- (N, 13, 13, 255)
- (N, 26, 26, 255)
- (N, 52, 52, 255)

In YoloV3 the original input image will be divided into (13, 13),  (26, 26) and (52, 52) grids as below.

![grids](https://github.com/Qucy/yolo3-tf2/blob/master/img/grids.jpg)

And for each grid will have 3 pre-defined anchors as below(white boxes), model will predict whether current grid contains an object's center point, the categorical for this object and x offset, y offset, width and height for bounding box.

![anchor_boxes](https://github.com/Qucy/yolo3-tf2/blob/master/img/anchor_boxes.jpg)

#### 3.2 Decode prediction result

Because Yolov3 is using 3 pre-defined anchors. so we can reshape our prediction result as below:

- (N, 13, 13, 255) -> (N, 13, 13, 3, 85)
- (N, 26, 26, 255) -> (N, 26, 26, 3, 85)
- (N, 52, 52, 255) -> (N, 26, 26, 3, 85)

And 85 can be divide into 4 + 1 + 80:

- 4 - stands for the x offset, y offset, width and height
- 1 - stands for the confidence score whether there is an object in the anchor box,  1 means yes, 0 means no.
- 80 - stands for number of classes in current dataset

In general decode have 2 steps:

- First adding x offset and y offset to grid start coordinates to get center point of predicted bounding box
- Then scale pre-defined anchor box width and height to get predicted width and height for bounding box

More detailed is explained as below image, **tx** and **ty** is predicted x offset and y offset, **tw** and **th** is predicted width and height need to be scaled. First retrieve top left coordinates (**cx**, **cy**) and adding **sigmoid tx** and **sigmoid ty** to get predicted center point **(bx**, **by**). And then use Exponential function to calculate predicted width(**bw**) and height (**bh**).

![decode_anchors](https://github.com/Qucy/yolo3-tf2/blob/master/img/decode_anchors.png)

And below is the code for decoding prediction into bounding boxes coordinates:

```python
import tensorflow as tf
from tensorflow.keras import backend as K


#-----------------------------------------------------#
#adjust with box coordinates to match the original image
#-----------------------------------------------------#
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    #-----------------------------------------------------------------#
    # revers y ans h to first dimension
    #-----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    if letterbox_image:
        new_shape = K.round(image_shape * K.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = K.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]])
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

#---------------------------------------------------#
#   Adjust predicted result to align with original image
#---------------------------------------------------#
def get_anchors_and_decode(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    #------------------------------------------#
    #   grid_shape = (width, height) = (13, 13) or (26, 26) or (52, 52)
    #------------------------------------------#
    grid_shape = K.shape(feats)[1:3]
    #--------------------------------------------------------------------#
    #   generate grip with shape => (13, 13, num_anchors, 2) => by default (13, 13, 3, 2)
    #--------------------------------------------------------------------#
    grid_x  = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
    grid_y  = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
    grid    = K.cast(K.concatenate([grid_x, grid_y]), K.dtype(feats))
    #---------------------------------------------------------------#
    #   adjust pre-defined anchors to shape (13, 13, num_anchors, 2)
    #---------------------------------------------------------------#
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, num_anchors, 2])
    anchors_tensor = K.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1])

    #---------------------------------------------------#
    #   reshape prediction results to (batch_size,13,13,3,85)
    #   85 = 4 + 1 + 80
    #   4 -> x offset, y offset, width and height
    #   1 -> confidence score
    #   80 -> 80 classes
    #---------------------------------------------------#
    feats           = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    #------------------------------------------#
    #   calculate bounding box center point bx, by, width(bw), height(bh) and normalized by grid shape (13, 26 or 52)
    #   bx = sigmoid(tx) + cx
    #   by = sigmoid(tx) + cy
    #   bw = pw * exp(tw)
    #   bh = ph * exp(th)
    #------------------------------------------#
    box_xy          = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh          = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    #------------------------------------------#
    #   retrieve confidence score and class probs
    #------------------------------------------#
    box_confidence  = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    #---------------------------------------------------------------------#
    #   if calc loss return -> grid, feats, box_xy, box_wh
    #   if during prediction return -> box_xy, box_wh, box_confidence, box_class_probs
    #---------------------------------------------------------------------#
    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

#---------------------------------------------------#
#   decode model outputs and return
#   1 - box coordinates (x1, y1, x2, y2)
#   2 - confidence score
#   3 - classes score
#---------------------------------------------------#
def DecodeBox(outputs,    # outputs from YoloV3
            anchors,      # pre-defined anchors in configuration
            num_classes,  # COCO=80, VOC=20
            input_shape,  # image shape 416 * 416
            #-----------------------------------------------------------#
            #   13x13's anchor are [116,90],[156,198],[373,326]
            #   26x26's anchors are [30,61],[62,45],[59,119]
            #   52x52's anchors are [10,13],[16,30],[33,23]
            #-----------------------------------------------------------#
            anchor_mask     = [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            max_boxes       = 100,
            confidence      = 0.5,
            nms_iou         = 0.3,
            letterbox_image = True):
    # reshape
    image_shape = K.reshape(outputs[-1],[-1])

    box_xy = []
    box_wh = []
    box_confidence  = []
    box_class_probs = []
    # loop number of pre-defined anchors (by default is 3)
    for i in range(len(anchor_mask)):
        sub_box_xy, sub_box_wh, sub_box_confidence, sub_box_class_probs = \
            get_anchors_and_decode(outputs[i], anchors[anchor_mask[i]], num_classes, input_shape)
        box_xy.append(K.reshape(sub_box_xy, [-1, 2]))
        box_wh.append(K.reshape(sub_box_wh, [-1, 2]))
        box_confidence.append(K.reshape(sub_box_confidence, [-1, 1]))
        box_class_probs.append(K.reshape(sub_box_class_probs, [-1, num_classes]))
    box_xy          = K.concatenate(box_xy, axis = 0)
    box_wh          = K.concatenate(box_wh, axis = 0)
    box_confidence  = K.concatenate(box_confidence, axis = 0)
    box_class_probs = K.concatenate(box_class_probs, axis = 0)

    #------------------------------------------------------------------------------------------------------------#
    #   Before image pass into Yolo network there is a pre-process method letter_box_image will padding gray points around
    #   image if size is not enough. So predicted box_xy, box_wh need to be adjusted to align with previous image and convert
    #   to Xmin, Ymin and Xmax, Ymax format.
    #   If model skip letterbox_image pre-process method, here still need to scale up to align with original image due to normalization.
    #------------------------------------------------------------------------------------------------------------#
    boxes       = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

    box_scores  = box_confidence * box_class_probs

    #-----------------------------------------------------------#
    #   is box score greater than score threshold
    #-----------------------------------------------------------#
    mask             = box_scores >= confidence
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_out   = []
    scores_out  = []
    classes_out = []
    for c in range(num_classes):
        #-----------------------------------------------------------#
        #   retrieve all the boxes and box scores >= score threshold
        #-----------------------------------------------------------#
        class_boxes      = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        #-----------------------------------------------------------#
        #   retrieve NMS index via IOU threshold
        #-----------------------------------------------------------#
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=nms_iou)

        #-----------------------------------------------------------#
        #   retrieve boxes, boxes scores and classes via NMS index
        #-----------------------------------------------------------#
        class_boxes         = K.gather(class_boxes, nms_index)
        class_box_scores    = K.gather(class_box_scores, nms_index)
        classes             = K.ones_like(class_box_scores, 'int32') * c

        boxes_out.append(class_boxes)
        scores_out.append(class_box_scores)
        classes_out.append(classes)
    boxes_out      = K.concatenate(boxes_out, axis=0)
    scores_out     = K.concatenate(scores_out, axis=0)
    classes_out    = K.concatenate(classes_out, axis=0)

    return boxes_out, scores_out, classes_out
```

#### 3.3 Confidence threshold and Non-Maximum-suppression

After decode prediction result, we still need to:

- select good bounding box only if confidence score is greater than confidence threshold
- select best bounding box with highest confidence score among in the same bounding boxes.

Below is an example for a image before and after Non-Maximum suppression

![nms](https://github.com/Qucy/yolo3-tf2/blob/master/img/nms.jpg)


### 4. Loss function

Before model training we need to define loss function to training our model and loss function is basically calculate difference between y true and y predict. For yoloV3 prediction is the outputs from model, for ground truth is the real bounding box coordinates in the real image. Both prediction and ground truth need to be encode or decoded before pass to loss function. And after encoding or decoding y true and y predict should have same shape as below:

- (batch_size, 13, 13, 3, 85)
- (batch_size, 26, 26, 3, 85)
- (batch_size, 52, 52, 3, 85)

#### 4.1 Encode Ground truth

The ground truth we get from annotation file are corners points for bounding box in the original image, which is (x1,y1) for the top left corner and (x2, y2) for the right bottom corner.

- First calculate center point, width and heights for true bounding box. Then divide by input shape for normalization.
- Find the best pre-defined anchor box for ground truth and record it

Below is the source code for encode ground truth:

```python
def preprocess_true_boxes(self, true_boxes, input_shape, anchors, num_classes):
    """
    preprocess true boxes
    :param true_boxes: ground truth boxes with shape (m, n, 5)
                       m: stands for number of images
                       n: stands for number of boxes
                       5: stands for x_min, y_min, x_max, y_max and class_id
    :param input_shape: 416*416
    :param anchors: size of pre-defined 9 anchor boxes
    :param num_classes: number of classes
    :return:
    """
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'

    true_boxes  = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')

    #-----------------------------------------------------------#
    #   3 feature layers in total
    #-----------------------------------------------------------#
    num_layers  = len(self.anchors_mask)
    #-----------------------------------------------------------#
    #   m -> number of images，grid_shapes -> [[13,13], [26,26], [52,52]]
    #-----------------------------------------------------------#
    m           = true_boxes.shape[0]
    grid_shapes = [input_shape // {0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    #-----------------------------------------------------------#
    #   y_true -> [(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)]
    #-----------------------------------------------------------#
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(self.anchors_mask[l]), 5 + num_classes),
                dtype='float32') for l in range(num_layers)]

    #-----------------------------------------------------------#
    #   calculate center point xy, box width and box height
    #   boxes_xy shape -> (m,n,2)  boxes_wh -> (m,n,2)
    #-----------------------------------------------------------#
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh =  true_boxes[..., 2:4] - true_boxes[..., 0:2]
    #-----------------------------------------------------------#
    #   normalization
    #-----------------------------------------------------------#
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    #-----------------------------------------------------------#
    #   [9,2] -> [1,9,2]
    #-----------------------------------------------------------#
    anchors         = np.expand_dims(anchors, 0)
    anchor_maxes    = anchors / 2.
    anchor_mins     = -anchor_maxes

    #-----------------------------------------------------------#
    #   only retrieve image width > 0
    #-----------------------------------------------------------#
    valid_mask = boxes_wh[..., 0]>0

    # loop all the image
    for b in range(m):
        #-----------------------------------------------------------#
        #   only retrieve image width > 0
        #-----------------------------------------------------------#
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        #-----------------------------------------------------------#
        #   [n,2] -> [n,1,2]
        #-----------------------------------------------------------#
        wh          = np.expand_dims(wh, -2)
        box_maxes   = wh / 2.
        box_mins    = - box_maxes

        #-----------------------------------------------------------#
        #   Calculate IOU between true box and pre-defined anchors
        #   intersect_area  [n,9]
        #   box_area        [n,1]
        #   anchor_area     [1,9]
        #   iou             [n,9]
        #-----------------------------------------------------------#
        intersect_mins  = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh    = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]

        box_area    = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]

        iou = intersect_area / (box_area + anchor_area - intersect_area)
        best_anchor = np.argmax(iou, axis=-1)

        # loop all the best anchors, try to find it to which feature layer below
        # (m 13, 13, 3, 85), (m 26, 26, 3, 85),  (m 52, 52, 3, 85)
        for t, n in enumerate(best_anchor):
            #-----------------------------------------------------------#
            #   Loop all the layers
            #-----------------------------------------------------------#
            for l in range(num_layers):
                if n in self.anchors_mask[l]:
                    #-----------------------------------------------------------#
                    #   using floor true boxes' x、y coordinates
                    #-----------------------------------------------------------#
                    i = np.floor(true_boxes[b,t,0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1] * grid_shapes[l][0]).astype('int32')
                    #-----------------------------------------------------------#
                    #   k -> index of pre-defined anchors
                    #-----------------------------------------------------------#
                    k = self.anchors_mask[l].index(n)
                    #-----------------------------------------------------------#
                    #   c -> the object category
                    #-----------------------------------------------------------#
                    c = true_boxes[b, t, 4].astype('int32')
                    #-----------------------------------------------------------#
                    #   y_true => shape => (m,13,13,3,85) or (m,26,26,3,85) or (m,52,52,3,85)
                    #-----------------------------------------------------------#
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true
```



#### 4.2 Calculate loss

After decode prediction result and encode ground truth result then we can start to calculate loss for our model and in YoloV3 the loss can be divided into 4 parts

- For true positive, the difference between predicted bounding box coordinates and true bounding box coordinates, including (x, y) and (w, h)
- For true positive, the difference between predicted object confidence score and 1
- For true positive, the cross entropy loss for object classes



### 5. Train your model

#### 5.1 Prepare your dataset

To train your model you need prepare your datasets first, you can use VOC datasets or COCO datasets to train your model.

For VOC datasets you can download here http://host.robots.ox.ac.uk/pascal/VOC/

For COCO datasets you can download here https://cocodataset.org/#download

But for your own data you need to install a image label tool to label your data first:

You can use pip to install LabelImage and label your own image. link -> https://pypi.org/project/labelImg/



#### 5.2 Preprocess your dataset

Before to train your model, we need to preprocess our dataset, since the the VOC or COCO dataset's annotation is in XML format. We need to process it via **voc_annotation.py**.  Change your dataset path accordingly and change annotation_mode = 2 to generate train and validation dataset.

After preprocess your dataset successfully you should see 2007_train.txt and 2007_val.txt.



#### 5.3 Train your model

By using **voc_annotation.py** we've generated our training and testing datasets. By point our train path to these 2 files and we run train.py file to kickoff the training. Of course you can change the hyper parameter in the train.py and the model weights will be saved in logs file every epoch.



#### 5.4 Make predictions !

After your model is trained, you can modify the model weights file path point to the latest weights file path in the logs folder. And input the image path or folder and run predict.py to trigger the prediction.