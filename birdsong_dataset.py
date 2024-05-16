import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import librosa
import noisereduce as nr
from scipy.io import wavfile

counter = 0
class SoundData():
    def __init__(self,df , path = "data/songs" , transform = None , max_len = 2, overlapping = 0.0, noise_reduction = False , noise_reduction_level = 0.5):


        self.transform =  transform
        self.max_len = max_len
        self.overlapping = overlapping
        self.noise_reduction = noise_reduction
        self.noise_reduction_level = noise_reduction_level
        self.path = Path.cwd().joinpath(path)
        self.metadata = df

        a = np.arange(0, len(self.metadata))
        np.random.shuffle(a)
        n = len(self.metadata)
        perc_train, per_val = 1 , 0

        self.train_idx, self.val_idx, self.test_idx = a[:int(n * perc_train)], a[int(n * perc_train):int(n * (perc_train+per_val))], a[int(n * (perc_train + per_val)):]

        self.train_samples , self.val_samples , self.test_samples = [] , [] ,[]

        self.names = self.encode_names()
        self.species = self.encode_species()
        self.load_data()

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
        assert (self.overlapping >= 0  and self.overlapping < 1 ), "invalid input"
        print("Loading The Data...")

        for idx in range(len(self.metadata)):
            file_name = "xc" + str(self.metadata.iloc[idx].file_id) + ".flac"
            audio_path = self.path.joinpath(file_name)
            audio, hz = librosa.load(audio_path)
            if self.noise_reduction:
                audio = self.reduce_noise(audio,hz)
            start = 0
            while start + self.max_len * hz < len(audio):
                sub_audio = audio[start : start + self.max_len * hz]
                mel_spec = self.convert_to_mel_spectrogram(sub_audio,hz)
                mel_spec = np.array(mel_spec).reshape((mel_spec.shape[0], mel_spec.shape[1], 1))
                if self.transform is not None:
                    mel_spec = self.transform(mel_spec)
                sample = [ mel_spec , self.names[idx], self.species[idx]]


                start += int(self.max_len * hz // (1 - self.overlapping))

                if idx in self.train_idx:
                    self.train_samples.append(sample)
                if idx in self.val_idx:
                    self.val_samples.append(sample)
                if idx in self.test_idx:
                    self.test_samples.append(sample)


        print("Data Loaded !")

    def convert_to_mel_spectrogram(self, audio , hz):
        sgram = librosa.stft(audio , hop_length=512 , win_length=1024)

        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=hz)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
        return mel_sgram
    @staticmethod
    def visualize_melspectrogram(sample):
        librosa.display.specshow(sample[0], sr=sample[1], x_axis='time', y_axis='mel', hop_length=512)

    def reduce_noise(self, audio , sr , name = None):

        global counter
        # print("NOISE REDUCTICTON ")
        reduced_noise = nr.reduce_noise(y=audio, sr=sr, prop_decrease = self.noise_reduction_level)

        if name is not None:
            wavfile.write(f"data/reduced/{name}.wav", sr, reduced_noise)
            counter += 1
        return reduced_noise
    @property
    def test(self):
        images = []
        labels_names = []
        labels_species = []

        for data in self.test_samples:
            images.append(data[0])
            labels_names.append( np.array(data[1]))
            labels_species.append( np.array(data[2]))


        images = np.array(images)  # transform to torch tensor
        names = np.array(labels_names)
        species = np.array(labels_species)
        return {"data" : images , "name_labels" : names , "species_labels" : species}
    @property
    def train(self):
        images = []
        labels_names = []
        labels_species = []

        for data in self.train_samples:
            images.append(data[0])
            labels_names.append( np.array(data[1]))
            labels_species.append( np.array(data[2]))


        images = np.array(images)  # transform to torch tensor
        names = np.array(labels_names)
        species = np.array(labels_species)
        return {"data" : images , "name_labels" : names , "species_labels" : species}
    @property
    def val(self):
        images = []
        labels_names = []
        labels_species = []

        for data in self.val_samples:
            images.append(data[0])
            labels_names.append( np.array(data[1]))
            labels_species.append( np.array(data[2]))


        images = np.array(images)  # transform to torch tensor
        names = np.array(labels_names)
        species = np.array(labels_species)
        return {"data" : images , "name_labels" : names , "species_labels" : species}


#




class AudioDataset(Dataset):
    def __init__(self , sounds ,transform = None):
        # super(AudioDataset,self).__init__()
        self.transform = transform
        self.data = sounds['data']
        self.names = sounds['name_labels']
        self.species = sounds['species_labels']
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform([self.data[idx] , self.names[idx] , self.species[idx]])
        return self.data[idx] , self.names[idx] , self.species[idx]

