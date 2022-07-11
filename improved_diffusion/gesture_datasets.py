from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


import os
import numpy as np
import datetime

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader

def load_data(
    *, data_dir, batch_size, motion_features
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    #all_files = _list_image_files_recursively(data_dir)
    #classes = None
    #if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        #class_names = [bf.basename(path).split("_")[0] for path in all_files]
        #sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        #classes = [sorted_classes[x] for x in class_names]
    dataset = GestureDataset(
        os.path.join(data_dir, 'input'),
        os.path.join(data_dir, 'label'),
        motion_features
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    )
    while True:
        yield from loader

class GestureDataset(Dataset):
    def __init__(self, audio_path, motion_path, motion_features):
        """
        Args:
            audio_path, motion_path: absolute path to audio and motion
        """
        def sortfiles(name):
            name = int(name.split('_')[-1].split('.')[0])
            return name
        # Lista as pastas que contem as amostras de cada take
        self.audios_path = [int(i) for i in os.listdir(audio_path)]
        self.audios_path.sort()
        # Cria o caminho completo para essas pastas que contem as amostras de cada take
        self.audios_path = [os.path.join(audio_path, str(i)) for i in self.audios_path]
        # Cria uma lista de amostras (sorted) para cada pasta
        self.audio_samples_per_folder = []
        for audio_folder in self.audios_path:
            folder_list = os.listdir(audio_folder)
            folder_list.sort(key=sortfiles)
            self.audio_samples_per_folder.append(folder_list)
        # Verifica quantas amostras cada pasta (take) possui
        self.n_audio_samples = [len(audio_folder) for audio_folder in self.audio_samples_per_folder]
        # Acumula a quantidade de amostras
        self.n_cumulative_audio_samples = [np.sum(self.n_audio_samples[:i+1]) for i in range(len(self.n_audio_samples))]

        # Faz o mesmo para os movimentos
        self.motions_path = [int(i) for i in os.listdir(motion_path)]
        self.motions_path.sort()
        self.motions_path = [os.path.join(motion_path, str(i)) for i in self.motions_path]
        self.motion_samples_per_folder = []
        for motion_folder in self.motions_path:
            folder_list = os.listdir(motion_folder)
            folder_list.sort(key=sortfiles)
            self.motion_samples_per_folder.append(folder_list)
        self.n_motion_samples = [len(motion_folder) for motion_folder in self.motion_samples_per_folder]
        self.n_cumulative_motion_samples = [np.sum(self.n_motion_samples[:i+1]) for i in range(len(self.n_motion_samples))]

        #Check if the number of frames are equal
        assert self.n_cumulative_audio_samples[-1] == self.n_cumulative_motion_samples[-1]

        motion = np.load( os.path.join(self.motions_path[0], os.listdir(self.motions_path[0])[0] ))
        audio = np.load( os.path.join(self.audios_path[0] ,os.listdir(self.audios_path[0])[0]) )

        motion = motion/np.pi

        self.audio_features = audio.shape[1]
        #self.motion_features = motion.shape[1]
        self.motion_features = motion_features


    def __len__(self):
        return self.n_cumulative_audio_samples[-1]

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.n_cumulative_audio_samples, idx+1, side='left')
        if file_idx > 0:
            file_sub_idx = idx - self.n_cumulative_audio_samples[file_idx-1]
        else:
            file_sub_idx = idx

        audio = np.load( os.path.join(self.audios_path[file_idx],self.audio_samples_per_folder[file_idx][file_sub_idx]) ).astype(np.float32)
        motion = np.load( os.path.join(self.motions_path[file_idx],self.motion_samples_per_folder[file_idx][file_sub_idx]) ).astype(np.float32)[:,:self.motion_features]
        out_dict = {}
        out_dict['in_audio'] = audio

        return motion, out_dict

    def n_channels(self):
        #return self.audio_features, self.motion_features #26,498
        return self.audio_features, self.motion_features #26,498
