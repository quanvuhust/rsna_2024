import torch
import torch.nn as nn
import torch.nn.functional as F

class ObjectDetectionLoss(nn.Module):
    def __init__(self, num_classes=5, no_object_class_id=5):
        super(ObjectDetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.no_object_class_id = no_object_class_id
        # Loss functions for classification and bounding boxes
        self.classification_loss_fn = nn.CrossEntropyLoss()
        self.bbox_loss_fn = nn.SmoothL1Loss()

    def forward(self, pred_class_logits, pred_bboxes, gt_classes, gt_bboxes, gt_object_mask):
        """
        Args:
            pred_class_logits: Predicted class scores for each object, shape (batch_size, max_objects, num_classes + 1)
            pred_bboxes: Predicted bounding boxes, shape (batch_size, max_objects, 4)
            gt_classes: Ground truth classes, shape (batch_size, max_objects)
            gt_bboxes: Ground truth bounding boxes, shape (batch_size, max_objects, 4)
            gt_object_mask: Binary mask (1 for objects, 0 for no object), shape (batch_size, max_objects)
        
        Returns:
            total_loss: The combined classification and bounding box regression loss.
        """
        batch_size, max_objects, _ = pred_class_logits.shape
        
        # Classification loss: Compare predicted classes to ground truth classes
        # We use the object mask to avoid penalizing predictions where no object is present
        cls_loss = self.classification_loss_fn(
            pred_class_logits.view(-1, self.num_classes + 1),  # Flatten for batch processing
            gt_classes.view(-1)  # Ground truth classes
        )

        # Bounding box loss: Only calculate the loss for objects, skip the no-object predictions
        bbox_loss = self.bbox_loss_fn(
            pred_bboxes[gt_object_mask == 1],  # Only consider predicted bounding boxes where objects exist
            gt_bboxes[gt_object_mask == 1]     # Ground truth bounding boxes for objects
        )

        # You could scale the losses to give different weight to classification and bounding box losses
        total_loss = cls_loss + bbox_loss
        return total_loss