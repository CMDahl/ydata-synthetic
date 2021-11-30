"PATEGAN implementation supporting Differential Privacy budget specification."
from typing import List

# pylint: disable = W0622, E0401
import tqdm
from tensorflow import clip_by_value
from tensorflow.dtypes import cast, float64
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.math import abs, exp, pow, reduce_sum, square
from tensorflow.keras.losses import BinaryCrossentropy

from ydata_synthetic.synthesizers.gan import BaseModel


class PATEGAN(BaseModel):
    "A basic PATEGAN synthesizer implementation with configurable differential privacy budget."

    __MODEL__='PATEGAN'

    def __init__(self, model_parameters, n_teachers: int, delta: float, epsilon: float):
        super().__init__(model_parameters)
        self.n_teachers = n_teachers
        self.delta = delta
        self.epsilon = epsilon

    def define_gan(self):
        def discriminator():
            discriminator = Discriminator(self.batch_size)
            return discriminator.build_model((self.data_dim,), self.layers_dim)

        self.generator = Generator(self.batch_size). \
            build_model(input_shape=(self.noise_dim,), dim=self.layers_dim, data_dim=self.data_dim)
        self.s_discriminator = discriminator()
        self.t_discriminators = [discriminator() for i in range(self.n_teachers)]

        generator_optimizer = Adam(learning_rate=self.g_lr)
        discriminator_optimizer = Adam(learning_rate=self.d_lr)

        loss_fn = BinaryCrossentropy(from_logits=True)
        self.generator.compile(loss=loss_fn, optimizer=generator_optimizer)
        self.s_discriminator.compile(loss=loss_fn, optimizer=discriminator_optimizer)
        for teacher in self.t_discriminators:
            teacher.compile(loss=loss_fn, optimizer=discriminator_optimizer)

    # pylint: disable = C0103
    def _moments_acc(self, votes, lap_scale, l_list):
        q = (2 + lap_scale * abs(2 * votes - self.n_teachers))/(4 * exp(lap_scale * abs(2 * votes - self.n_teachers)))

        update = []
        for l in l_list:
            clip = 2 * square(lap_scale) * l * (l + 1)
            t = (1 - q) * pow((1 - q) / (1 - exp(2*lap_scale) * q), l) + q * exp(2 * lap_scale * l)
            update.append(reduce_sum(clip_by_value(t, clip_value_min=-clip, clip_value_max=clip)))
        return cast(update, dtype=float64)

    def train(self, data, train_arguments, num_cols: List[str], cat_cols: List[str],
              preprocess: bool = True):
        return None


class Discriminator(Model):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_model(self, input_shape, dim):
        input = Input(shape=input_shape, batch_size=self.batch_size)
        x = Dense(dim * 4)(input)
        x = ReLU()(x)
        x = Dense(dim * 2)(x)
        x = Dense(1)(x)
        return Model(inputs=input, outputs=x)


class Generator(Model):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_model(self, input_shape, dim, data_dim):
        input = Input(shape=input_shape, batch_size = self.batch_size)
        x = Dense(dim)(input)
        x = ReLU()(x)
        x = Dense(dim * 2)(x)
        x = Dense(data_dim)(x)
        return Model(inputs=input, outputs=x)
