import os
from flow_style_gestures import visualization
from flow_style_gestures import motion
import numpy as np
import datetime

from torch.utils.data import DataLoader, Dataset
from flow_style_gestures.glow.builder import build
from flow_style_gestures.glow.trainer import Trainer
from flow_style_gestures.glow.generator import Generator
from flow_style_gestures.glow.config import JsonConfig
from torch.utils.data import DataLoader

class GestureDataset(Dataset):
    def __init__(self, audio_path, motion_path, step=30):
        """
        Args:
            audio_path, motion_path: absolute path to audio and motion
        """
        self.step = step
        self.audios_path = [int(i) for i in os.listdir(audio_path)]
        self.audios_path.sort()
        self.audios_path = [os.path.join(audio_path, str(i)) for i in self.audios_path]
        self.n_audio_samples = [len(os.listdir(audio_folder)) for audio_folder in self.audios_path]
        self.n_cumulative_audio_samples = [np.sum(self.n_audio_samples[:i+1]) for i in range(len(self.n_audio_samples))]

        self.motions_path = [int(i) for i in os.listdir(motion_path)]
        self.motions_path.sort()
        self.motions_path = [os.path.join(motion_path, str(i)) for i in self.motions_path]
        self.n_motion_samples = [len(os.listdir(motion_folder)) for motion_folder in self.motions_path]
        self.n_cumulative_motion_samples = [np.sum(self.n_motion_samples[:i+1]) for i in range(len(self.n_motion_samples))]

        #Check if the number of frames are equal
        assert self.n_cumulative_audio_samples[-1] == self.n_cumulative_motion_samples[-1]

        audio = np.load( os.path.join(self.audios_path[0] ,os.listdir(self.audios_path[0])[0]) )
        motion = np.load( os.path.join(self.motions_path[0], os.listdir(self.motions_path[0])[0] ))

        self.audio_features = audio.shape[1]
        self.motion_features = motion.shape[1]


    def __len__(self):
        return self.n_cumulative_audio_samples[-1]//self.step

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.n_cumulative_audio_samples, idx*self.step+1, side='left')
        if file_idx > 0:
            file_sub_idx = idx*self.step - self.n_cumulative_audio_samples[file_idx-1]
        else:
            file_sub_idx = idx*self.step
        #print('idx: %i\nfile_idx: %i\nfile_sub_idx: %i' % (idx, file_idx, file_sub_idx))

        if idx >= self.__len__():
            print('Tem problema aqui. idx = %i, len = %i\nultimo arquivo:%s' % (idx*self.step, self.__len__(), self.audios_path[-1]))
        if file_idx >= len(self.audios_path):
            print('Tem problema aqui 2. idx = %i, file_idx = %i, len = %i\nultimo arquivo:%s' % (idx*self.step, file_idx, len(self.audios_path), self.audios_path[-1]))
        if file_sub_idx >= len(os.listdir(self.audios_path[file_idx])):
            print('Tem problema aqui 3. idx = %i, file_idx = %i, file_sub_idx = %i, len = %i\nultimo arquivo:%s' % (idx*self.step, file_idx,file_sub_idx, len(os.listdir(self.audios_path[file_idx])), self.audios_path[-1]))

        audio = np.load( os.path.join(self.audios_path[file_idx] ,os.listdir(self.audios_path[file_idx])[file_sub_idx]) ).astype(np.float32)
        motion = np.load( os.path.join(self.motions_path[file_idx], os.listdir(self.motions_path[file_idx])[file_sub_idx] )).astype(np.float32)

        audio = np.swapaxes(audio, 0, 1)
        motion = np.swapaxes(motion, 0, 1)


        sample = {'x': audio, 'cond': motion}
        return sample

    def n_channels(self):
        return self.audio_features, self.motion_features #26,498

class Genea2022():
    def __init__(self, audio_path, motion_path, val_audio_path, val_motion_path, tst_audio_path, tst_motion_path):
        self.train_dataset = GestureDataset(audio_path, motion_path)
        self.validation_dataset = GestureDataset(val_audio_path, val_motion_path)
        self.test_dataset = GestureDataset(tst_audio_path, tst_motion_path)

    def n_channels(self):
        return self.train_dataset.n_channels()

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_validation_dataset(self):
        return self.validation_dataset


params_path = 'flow_style_gestures/hparams/preferred/genea2022.json'
audio_path = "D:\\genea_old\\dataset_v1\\trn\\processed\\input"
motion_path = "D:\\genea_old\\dataset_v1\\trn\\processed\\label"
val_audio_path = "D:\\genea_old\\dataset_v1\\val\\processed\\input"
val_motion_path = "D:\\genea_old\\dataset_v1\\val\\processed\\label"
tst_audio_path = "D:\\genea_old\\dataset_v1\\tst\\processed_val\\input"
tst_motion_path = "D:\\genea_old\\dataset_v1\\tst\\processed_val\\label"
dataset = Genea2022(audio_path, motion_path, val_audio_path, val_motion_path, tst_audio_path, tst_motion_path)
hparams = JsonConfig(params_path)

date = str(datetime.datetime.now())
date = date[:date.rfind(":")].replace("-", "")\
                             .replace(":", "")\
                             .replace(" ", "_")
log_dir = os.path.join(hparams.Dir.log_root, "log_" + date)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

print("log_dir:" + str(log_dir))

is_training = hparams.Infer.pre_trained == ""

x_channels, cond_channels = dataset.n_channels()
is_training = True
# build graph
built = build(x_channels, cond_channels, hparams, is_training)

if is_training:
    # build trainer
    trainer = Trainer(**built, data=dataset, log_dir=log_dir, hparams=hparams)

# train model
trainer.train()
