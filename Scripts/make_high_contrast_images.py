from os import listdir as ls
from os.path import join
from os import makedirs
from numpy import array
from tqdm import tqdm
import cv2


def high_contrast_image(img: array) -> array:
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


def to_grayscale(img: array) -> array:
    grayscale = cv2.cvtColor(img,
                             cv2.COLOR_BGR2GRAY)
    return grayscale


def mkdir(path: str) -> None:
    makedirs(path,
             exist_ok=True)


params = {
    "path data": "../Data",
    "path images": "normal",
    "path grayscale": "grayscale",
    "path high contrast": "high_contrast",
    "folders": [
        "test",
        "validate",
        "train",
    ]
}

for folder in params["folders"]:
    path = join(params["path data"],
                folder,
                params["path images"],
                "data")
    path_high_contrast = join(params["path data"],
                              folder,
                              params["path high contrast"],
                              "data")
    path_grayscale = join(params["path data"],
                          folder,
                          params["path grayscale"],
                          "data")
    mkdir(path_high_contrast)
    mkdir(path_grayscale)
    files = sorted(ls(path))
    for file in tqdm(files,
                     desc=f"{folder} images"):
        filename = join(path, file)
        image_original = cv2.imread(filename)
        image = high_contrast_image(image_original)
        grayscale = to_grayscale(image)
        filename = join(path_high_contrast,
                        file)
        cv2.imwrite(filename, image)
        filename = join(path_grayscale,
                        file)
        cv2.imwrite(filename,
                    grayscale)
