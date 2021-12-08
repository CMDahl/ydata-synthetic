"Activation Interface layer test suite."
from numpy import cumsum, isin, split
from numpy import sum as npsum
from numpy.random import normal
from pytest import fixture
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input

from ydata_synthetic.utils.gumbel_softmax import ActivationInterface


@fixture(name='noise_batch')
def fixture_noise_batch():
    "Sample noise for mock output generation."
    return normal(size=(10, 16))

@fixture(name='mock_col_map')
def fixture_mock_col_map():
    "Mock data processing column map (var blocks i/o names)."
    return {'numerical': [
        [f'nfeat{n}' for n in range(6)],
        [f'nfeat{n}' for n in range(6)]],
        'categorical': [
        [f'cfeat{n}' for n in range(2)],
        sum([[f'cfeat0_{i}' for i in range(4)], [f'cfeat1_{i}' for i in range(2)]],[])]}

# pylint: disable=C0103
@fixture(name='mock_generator')
def fixture_mock_generator(noise_batch, mock_col_map):
    "A mock generator with the Activation Interface as final layer."
    input_ = Input(shape=noise_batch.shape[1], batch_size = noise_batch.shape[0])
    dim = 15
    data_dim = 12
    x = Dense(dim, activation='relu')(input_)
    x = Dense(dim * 2, activation='relu')(x)
    x = Dense(dim * 4, activation='relu')(x)
    x = Dense(data_dim)(x)
    x = ActivationInterface(mock_col_map, name='act_itf')(x)
    return Model(inputs=input_, outputs=x)

@fixture(name='mock_output')
def fixture_mock_output(noise_batch, mock_generator):
    "Returns mock output of the model as a numpy object."
    return mock_generator(noise_batch).numpy()

# pylint: disable=W0632
def test_io(noise_batch, mock_col_map, mock_output):
    "Tests the output format of the activation interface for a known input."
    num_lens = len(mock_col_map.get('numerical')[1])
    cat_lens = len(mock_col_map.get('categorical')[1])
    assert mock_output.shape == (len(noise_batch), num_lens + cat_lens), "The output has wrong shape."
    num_part, cat_part = split(mock_output, [num_lens], 1)
    assert not isin(num_part, [0, 1]).all(), "The numerical block is not expected to contain 0 or 1."
    assert isin(cat_part, [0, 1]).all(), "The categorical block is expected to contain only 0 or 1."
    cat_i, cat_o = mock_col_map.get('categorical')
    cat_blocks = cumsum([len([col for col in cat_o if ''.join(col.split('_')[:-1]) == feat]) for feat in cat_i])
    cat_blocks = split(cat_part, cat_blocks[:-1], 1)
    assert all(npsum(abs(block)) == noise_batch.shape[0] for block in cat_blocks), "There are non one-hot encoded \
        categorical blocks."
