import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import librosa

#
class AudioDataset(Dataset):
    def __init__(self, path = "data/songs", metadata_path= "data/birdsong_metadata.csv" , transform = None):
        self.transform =  transform

        self.path = Path.cwd().joinpath(path)
        self.metadata = pd.read_csv(Path.cwd().joinpath(metadata_path))
        self.names = self.encode_names()
        self.species = self.encode_species()
        self.data = self.load_data()

    def __len__(self):
        return len(self.metadata)

    def encode_names(self):
        self.name_encoder = LabelEncoder()
        encoded_names = self.name_encoder.fit_transform(self.metadata.english_cname.values)
        return encoded_names

    def encode_species(self):
        self.species_encoder = LabelEncoder()
        encoded_species = self.species_encoder.fit_transform(self.metadata.species.values)
        return encoded_species

    def decode_name(self, id):
        return self.name_encoder.inverse_transform(id)[0]

    def decode_species(self, id):
        return self.species_encoder.inverse_transform(id)[0]

    def load_data(self):
        data = []
        print("Loading The Data...")
        for idx in range(len(self)):
            file_name = "xc" + str(self.metadata.iloc[idx].file_id) + ".flac"
            audio_path = self.path.joinpath(file_name)
            audio, hz = librosa.load(audio_path)
            mel_spec = self.convert_to_mel_spectrogram(audio,hz )
            sample  = [np.array(mel_spec).reshape((mel_spec.shape[0],mel_spec.shape[1],1)),  np.array(self.names[idx]), np.array(self.species[idx])]
            if self.transform is not None:
                sample = self.transform(sample)
            data.append(sample)
        print("Data Loaded !")

        return data

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1], self.data[idx][2]
    def convert_to_mel_spectrogram(self, audio , hz):
        sgram = librosa.stft(audio)
        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=hz)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
        return mel_sgram
    @staticmethod
    def visualize_melspectrogram(sample):
        librosa.display.specshow(sample[0], sr=sample[1], x_axis='time', y_axis='mel', hop_length=512)

#
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample[0]

        name = sample[1]
        species =sample[2]


        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image) , torch.from_numpy(name) , torch.from_numpy(species)

ds = AudioDataset(transform=ToTensor())
# print(ds[0])