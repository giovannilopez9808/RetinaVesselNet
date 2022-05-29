from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Nadam
from keras.applications import mobilenet_v2
from numpy import expand_dims, array, uint8
from argparse import Namespace
from pandas import DataFrame
from params import mkdir
from os.path import join
import tensorflow as tf


class unet_model:

    def __init__(self, unfreeze: str) -> None:
        output_classes = 1
        self._create_mobilenet(unfreeze)
        self.model = self._basic_unet_model(output_classes)

    def _create_mobilenet(self, unfreeze: str) -> None:
        base_model = mobilenet_v2.MobileNetV2(input_shape=[256, 256, 3],
                                              include_top=False)
        layer_names = [
            # 64x64
            'block_1_expand_relu',
            # 32x32
            'block_3_expand_relu',
            # 16x16
            'block_6_expand_relu',
            # 8x8
            'block_13_expand_relu',
            # 4x4
            'block_16_project',
        ]
        base_model_outputs = [
            base_model.get_layer(name).output for name in layer_names
        ]
        # Create the feature extraction model
        self.down_stack = tf.keras.Model(inputs=base_model.input,
                                         outputs=base_model_outputs)
        self.down_stack.trainable = False
        self._set_unfreeze(unfreeze)
        self.up_stack = [
            # 4x4 -> 8x8
            pix2pix.upsample(512, 3),
            # 8x8 -> 16x16
            pix2pix.upsample(256, 3),
            # 16x16 -> 32x32
            pix2pix.upsample(128, 3),
            # 32x32 -> 64x64
            pix2pix.upsample(64, 3),
        ]

    def _set_unfreeze(self, unfreeze: str) -> None:
        """
        Modos para descongelar parametros de la red de Mobile
        """
        if unfreeze == "all":
            self.down_stack.trainable = True
        if unfreeze == "last conv":
            for layer in self.down_stack.layers:
                if layer.name == "block_16_project":
                    layer.trainable = True
        if unfreeze == "none":
            pass

    def _basic_unet_model(self, output_channels: int) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=[256, 256, 3])
        # Downsampling through the model
        skips = self.down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])
        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])
        # This is the last layer of the model
        # 64x64 -> 128x128
        last = tf.keras.layers.Conv2DTranspose(filters=output_channels,
                                               kernel_size=3,
                                               strides=2,
                                               padding='same')
        x = last(x)
        x = Activation("sigmoid")(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

    def compile(self) -> None:
        """
        Compilacion del modelo
        """
        self.model.compile(optimizer=Nadam(learning_rate=1e-4),
                           loss="binary_crossentropy",
                           metrics=['accuracy'])

    def run(self, params: dict, args: Namespace, data_generator) -> DataFrame:
        """
        Ejecuccion del modelo realizando el guardado del historial y los pesos del modelo
        """
        filename = join(params["path results"], "checkpoint.pt")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filename,
                                                        save_weights_only=True,
                                                        monitor='val_accuracy',
                                                        mode='max',
                                                        save_best_only=True)
        self.history = self.model.fit(
            data_generator.train,
            validation_data=data_generator.validation,
            steps_per_epoch=args.steps_per_epoch_train,
            validation_steps=args.steps_per_epoch_val,
            epochs=args.epochs,
            callbacks=[checkpoint])
        self.save_history(params)
        self.save_model(params)

    def save_history(self, params: dict) -> None:
        """
        Guardado del historial del modelo
        """
        history = DataFrame(self.history.history)
        history.index.name = "Epoch"
        mkdir(params["path results"])
        filename = join(params["path results"], params["history name"])
        history.to_csv(filename)

    def save_model(self, params: dict) -> None:
        """
        Guardado del modelo
        """
        filename = join(params["path results"], params["model name"])
        self.model.save(filename)

    def summary(self) -> None:
        """
        Facil acceso al summary del modelo
        """
        self.model.summary()

    def load(self, filename: str) -> None:
        self.model.load_weights(filename)

    def predict(self, image: array) -> array:
        test_image = transform_image(image)
        result = self.model.predict(test_image)
        predict = (result[0, :, :, :] > 0.5).astype(uint8)
        return predict


def transform_image(image: array) -> array:
    test_image = expand_dims(image, 0)
    return test_image
