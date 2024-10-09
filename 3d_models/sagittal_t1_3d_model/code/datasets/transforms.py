import albumentations as A

def get_train_transform(image_size):
    normal_aug_list = [
                    A.Resize(height=image_size, width=image_size, p=1),
                    # A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.6, 1.0), p=1),
                    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.75),
                    # A.Perspective(p=0.75),
                    A.OneOf([
                        A.MotionBlur(blur_limit=5),
                        A.GaussNoise(var_limit=(5.0, 30.0)),
                    ], p=0.75),

                    # A.OneOf([
                    #     A.OpticalDistortion(distort_limit=1.0),
                    #     # A.GridDistortion(num_steps=5, distort_limit=1.),
                    #     A.ElasticTransform(alpha=3),
                    # ], p=0.75),
                    # A.HorizontalFlip(p=0.5),
                    # A.CoarseDropout(max_holes=4, max_height=int(288 * 0.1), max_width=int(288 * 0.1), p=0.5),
                    # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=35, border_mode=0, p=0.75),
                ]
    instance_aug_list = [
        A.Resize(height=image_size, width=image_size, p=1),
        # A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.7, 1.0), p=1),
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
        # A.Perspective(p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.5),
        # A.OneOf([
        #     A.OpticalDistortion(distort_limit=1.0),
        #     A.GridDistortion(num_steps=5, distort_limit=1.),
        #     # A.ElasticTransform(alpha=3),
        # ], p=0.5),
        # A.HorizontalFlip(p=0.5),
        # A.CoarseDropout(max_holes=1, max_height=int(288 * 0.1), max_width=int(288 * 0.1), p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=25, border_mode=0, p=0.5),
    ]
    return normal_aug_list, instance_aug_list