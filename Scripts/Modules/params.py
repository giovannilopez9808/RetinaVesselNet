from os import listdir, makedirs
from argparse import Namespace
from os.path import join


def get_params(select_data: str) -> dict:
    """
    parametros de los datos a tratar
    """
    params = {
        "root": "/content/drive/MyDrive/Patrones",
        # Direccion de los datos
        "path data": "Data",
        # Direccion original de los datos
        "folder image": "",
        # Direccion original de las mascaras
        "folder mask": "label",
        # Direccion de los resultados
        "path results": "",
    }
    if select_data == "high contrast":
        params["folder image"] = "high_contrast"
        params["path results"] = "Results/high_contrast"
    if select_data == "normal":
        params["folder image"] = "image"
        params["path results"] = "Results/normal"
    params["path data"] = join(params["root"],
                               params["path data"])
    params["path results"] = join(params["root"],
                                  params["path results"])
    return params


def get_args(params: dict) -> Namespace:
    """
    Definicion de los argumentos a utilizar en los entrenamientos
    """
    args = Namespace()
    args.train_len = len(params[f"train {params['folder image']}"])
    args.val_len = len(params[f"validate {params['folder image']}"])
    args.batch_size = 32
    args.steps_per_epoch = args.train_len // args.batch_size
    args.epochs = 15
    return args


def ls(path: str) -> list:
    """
    Estandarización del listdir
    """
    files = sorted(listdir(path))
    return files


def mkdir(path: str) -> None:
    """
    Estandarizacion del makedirs
    """
    makedirs(path,
             exist_ok=True)


def organization_files(params: dict, is_organized: bool = True) -> dict:
    """
    Organización de los datos obtenidos en la pagina
    https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

    params: diccionario con las direcciones de los datos
    """
    folders = [params["folder image"],
               params["folder mask"]]
    for folder in folders:
        for data_type in ["train", "test", "validate"]:
            # Definicion de las direcciones de los datos organizados
            path = join(params["path data"],
                        data_type,
                        folder)
            params[f"path {data_type} {folder} all"] = path
            path = join(path,
                        "data")
            params[f"path {data_type} {folder}"] = path
            params[f"{data_type} {folder}"] = ls(
                params[f"path {data_type} {folder}"])
    return params
