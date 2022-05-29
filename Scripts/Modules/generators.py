from keras.preprocessing.image import ImageDataGenerator
from numpy import where, expand_dims


class generator_model:

    def __init__(self, params, args):
        """
        Contiene los metodos para generar imagenes a partir de las encontradas en las rutas de entrenamiento y test

        Inputs:
        -----------------------
        params -> directorio con los diferentes parámetros a usar
            params["paths"] -> diccionario con las diferentes rutas de los archivos
                params["paths"]["train"] -> directorio con las diferentes rutas de datos de entrenamiento
                    params["paths"]["train"]["base"] -> ruta de los documentos de entrenamiento
                params["paths"]["test"] -> directorio con las diferentes rutas de los datos de test
                    params["paths"]["test"]["base"] -> ruta con los documentos de test
        """

        self.params = params
        self.args = args
        self._set_generators()
        self._get_data()
        self._get_validation()
        self._get_test()

    def _set_generators(self) -> dict:
        """
        Definicon de los generadores para los datos y las mascaras
        """
        self.image_data_args = dict(
            rotation_range=90,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.5,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='reflect',
        )
        self.mask_data_args = dict(rotation_range=90,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   shear_range=0.5,
                                   zoom_range=0.3,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='reflect',
                                   preprocessing_function=lambda x: where(
                                       x > 0, 1, 0).astype(x.dtype))

    def _get_data(self) -> None:
        """
        Creacion de los generadores de datos para los datos de entrenamiento y validacion
        """
        image_data_generator = ImageDataGenerator(**self.image_data_args,
                                                  rescale=1.0 / 255.0)
        self.train_image_generator = image_data_generator.flow_from_directory(
            self.params[f"path train {self.params['folder image']} all"],
            batch_size=self.args.batch_size,
            target_size=(256, 256),
            class_mode=None,
            seed=54)
        masks_data_generator = ImageDataGenerator(**self.mask_data_args)
        self.train_mask_generator = masks_data_generator.flow_from_directory(
            self.params[f"path train {self.params['folder mask']} all"],
            target_size=(256, 256),
            class_mode=None,
            seed=54,
            batch_size=self.args.batch_size)
        self.train = self.image_mask_generator(self.train_image_generator,
                                               self.train_mask_generator)

    def _get_validation(self) -> None:
        """
        Creacion de los generadores de datos para los datos de test
        """
        image_data_generator = ImageDataGenerator(rescale=1.0 / 255.0)
        self.validate_image_generator = image_data_generator.flow_from_directory(
            self.params[f"path validate {self.params['folder image']} all"],
            batch_size=40,
            seed=54,
            class_mode=None)
        self.validate_masks_generator = image_data_generator.flow_from_directory(
            self.params[f"path validate {self.params['folder mask']} all"],
            batch_size=40,
            seed=54,
            class_mode=None)
        self.validation = self.image_mask_generator(
            self.validate_image_generator, self.validate_masks_generator)

    def _get_test(self) -> None:
        """
        Creacion de los generadores de datos para los datos de test
        """
        image_data_generator = ImageDataGenerator(rescale=1.0 / 255.0)
        self.test_image_generator = image_data_generator.flow_from_directory(
            self.params[f"path test {self.params['folder image']} all"],
            batch_size=20,
            seed=54,
            class_mode=None)
        self.test_masks_generator = image_data_generator.flow_from_directory(
            self.params[f"path test {self.params['folder mask']} all"],
            batch_size=20,
            seed=54,
            class_mode=None)
        self.test = self.image_mask_generator(self.test_image_generator,
                                              self.test_masks_generator)

    def image_mask_generator(self, image_generator, mask_generator) -> tuple:
        train_generator = zip(image_generator, mask_generator)
        for (img, mask) in train_generator:
            mask = mask[:, :, :, 0]
            mask = expand_dims(mask, axis=3)
            # append
            yield (img, mask)
