import numpy as np
from torchvision import transforms
from PIL import Image


class setFlip(object):
    def __init__(self, flip):
        self.flip = flip

    def __call__(self, img):
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return(img)


class image_trian(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, flip, offset_x, offset_y):
        # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                                  std=[0.5, 0.5, 0.5])
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.functional.crop(
                img, top=offset_x, left=offset_y, height=img.size[0], width=img.size[1]),
            setFlip(flip),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        #img = transform(Image.fromarray(np.uint8(img)))
        return (img)


class landmark_transform(object):
    def __init__(self, img_size, flip_reflect):
        self.img_size = img_size
        self.flip_reflect = flip_reflect.astype(int) - 1

    def __call__(self, land, flip, offset_x, offset_y):

        land[:, 0] = land[:, 0] - offset_x
        land[:, 1] = land[:, 1] - offset_y
        # change the landmark orders when flipping
        if flip:
            land[:, 0] = self.img_size - 1 - land[:, 0]
            land[:, 0] = land[:, 0][self.flip_reflect]
            land[:, 1] = land[:, 1][self.flip_reflect]

        return (land)
