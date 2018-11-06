import layers
import keras
import soundfile as sf
import librosa
import resampy
import os
import numpy as np
import sys
from typing import NamedTuple

class MelGenerator(keras.utils.Sequence):
    def __init__(self, n_ids, batch_size, DATA_ROOT):
        self.n_ids = n_ids
        self.batch_size = batch_size
        self.mel_basedir = os.path.join(DATA_ROOT, 'melspec/')
        with np.load(os.path.join(DATA_ROOT, 'openmic-2018.npz')) as OPENMIC:
            self.Y_true, self.Y_mask, self.sample_key = OPENMIC['Y_true'], OPENMIC['Y_mask'], OPENMIC['sample_key']

    def __len__(self):
        return int(np.ceil(len(self.n_ids) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_n_ids = self.n_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self._data_from_ids(batch_n_ids)
    
    def _data_from_ids(self, n_ids):
        x_ids = self.sample_key[n_ids]
        x_batch = []
        y_batch = np.multiply(self.Y_true[n_ids], self.Y_mask[n_ids]) > 0.5
        for x_id in x_ids:
            mel_subdir = os.path.join(self.mel_basedir, x_id[:3])
            melpath = os.path.join(mel_subdir, x_id+'.npz')
            with np.load(melpath) as f:
                x_batch.append(f['melspec'].T)
        x_batch = np.array(x_batch)
        x_batch = np.expand_dims(x_batch, axis=-1)
        
        return x_batch.astype('float32'), y_batch
    
    def on_epoch_end(self):
        np.random.shuffle(self.n_ids)

class MelParam(NamedTuple):
    sr: int
    n_fft: int
    hop_length: int
        
def save_melspec(x_id, 
                 audio_basedir='/beegfs/qx244/ds/openmic-2018/audio/', 
                 mel_basedir='/beegfs/qx244/ds/openmic-2018/melspec/', 
                 recompute_if_exist=False,
                 param=MelParam(sr=44100,n_fft=2048,hop_length=512)):
    # create corresponding .npz files that stores the mel_spectrogram
    
    # check if feature dir exist
    audio_subdir = os.path.join(audio_basedir, x_id[:3])
    mel_subdir = os.path.join(mel_basedir, x_id[:3])
    if not os.path.isdir(mel_subdir):
        os.mkdir(mel_subdir)
    excerpt_path = os.path.join(audio_subdir, x_id+'.ogg')
    mel_path = os.path.join(mel_subdir, x_id+'.npz')
    
    if (not recompute_if_exist) and os.path.isfile(mel_path):
        print('{} already exist.'.format(mel_path))
        
    else:
        y, sr = sf.read(excerpt_path)
        # Make Mono
        if y.ndim == 2:
            y = y.mean(axis=-1)
        if sr != param.sr:
#             print('resampling excerpt {} from original sr {}'.format(x_id, sr))
            y = resampy.resample(y, sr, param.sr)
        melspec = librosa.feature.melspectrogram(y=y, sr=param.sr, 
                                                 n_fft=param.n_fft, hop_length=param.hop_length)
        if melspec.shape != (128,862):
            print("bad shape! ID:{}, shape: {}".format(x_id, melspec.shape))        
        np.savez(mel_path, x_id=x_id, melspec=melspec)

def construct_crnnL3_smp_tom(input_shape=(None, 128, 1), n_classes=20):
    '''
    Modified from Openmic's original models.py file
    CRNN with L3-style conv encoder
    Parameters
    ----------
    input_shape (defaults to (862, 128, 1).... but is this even what I think it is?)
    alpha (for softmax... large => max pooling)
    
    Returns
    -------
    Keras model
    '''
    
    input_feature = keras.Input(shape=input_shape)

    # Apply batch normalization
    x_bn = keras.layers.BatchNormalization()(input_feature)

    # BLOCK 1
    conv1 = keras.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = keras.layers.BatchNormalization()(conv1)
    conv2 = keras.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = keras.layers.BatchNormalization()(conv2)
    pool2 = keras.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = keras.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = keras.layers.BatchNormalization()(conv3)
    conv4 = keras.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = keras.layers.BatchNormalization()(conv4)
    pool4 = keras.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = keras.layers.Convolution2D(256, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = keras.layers.BatchNormalization()(conv5)
    conv6 = keras.layers.Convolution2D(256, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = keras.layers.BatchNormalization()(conv6)
    pool6 = keras.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = keras.layers.Convolution2D(512, (3, 3),
                                       padding='same',
                                       activation='relu',
                                       kernel_initializer='he_normal')(pool6)
    bn7 = keras.layers.BatchNormalization()(conv7)
    conv8 = keras.layers.Convolution2D(512, (3, 3),
                                       padding='same',
                                       activation='relu',
                                       kernel_initializer='he_normal')(bn7)
    bn8 = keras.layers.BatchNormalization()(conv8)
    pool8 = keras.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = keras.layers.Convolution2D(1024, (1, 8),
                                         padding='valid',
                                         activation='relu',
                                         kernel_initializer='he_normal')(pool8)
    bn8 = keras.layers.BatchNormalization()(conv_sq)
    sq2 = layers.SqueezeLayer(axis=-2)(bn8)

    # RNN
    # First recurrent layer: a 128-dim bidirectional gru
#     rnn1 = keras.layers.Bidirectional(keras.layers.GRU(128,
#                                                        return_sequences=True))(sq2)

    p0 = keras.layers.Dense(n_classes, activation='sigmoid')

    p_dynamic = keras.layers.TimeDistributed(p0, name='dynamic/tags')(sq2) #rnn1

    p_static = layers.AutoPool(axis=1,
                               name='static/tags')(p_dynamic)

    model = keras.models.Model(input_feature,
                               p_static)
    
    return model