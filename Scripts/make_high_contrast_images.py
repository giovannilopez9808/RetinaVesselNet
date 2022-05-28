from os import listdir as ls
from os.path import join
from tqdm import tqdm
import cv2


def high_contrast_image(img):
    """
    Filtro de alto contraste a la imagen
    """
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.,
                            tileGridSize=(8, 8))
    # convert from BGR to LAB color space
    lab = cv2.cvtColor(img,
                       cv2.COLOR_BGR2LAB)
    # split on 3 different channels
    l, a, b = cv2.split(lab)
    # apply CLAHE to the L-channel
    l2 = clahe.apply(l)
    # merge channels
    lab = cv2.merge((l2, a, b))
    # convert from LAB to BGR
    img = cv2.cvtColor(lab,
                       cv2.COLOR_LAB2BGR)
    return img


params = {
    "path data": "../Data",
    "folders": [
        "test",
        "validate",
        "train",
    ]
}

for folder in params["folders"]:
    path = join(params["path data"],
                folder,
                "image/data")
    path_out = join(params["path data"],
                    folder,
                    "high_contrast/data")
    files = sorted(ls(path))
    for file in tqdm(files):
        filename = join(path, file)
        image_original = cv2.imread(filename)
        image = high_contrast_image(image_original)
        filename = join(path_out,
                        file)
        cv2.imwrite(filename,
                    image)
