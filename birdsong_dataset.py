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
class AudioDataset(Dataset):
    def __init__(self, path = "data/songs", metadata_path= "data/birdsong_metadata.csv" , transform = None , max_len = 2, overlapping = 0.0, noise_reduction = False , noise_reduction_level = 0.5):


        self.transform =  transform
        self.max_len = max_len
        self.overlapping = overlapping
        self.noise_reduction = noise_reduction
        self.noise_reduction_level = noise_reduction_level
        self.path = Path.cwd().joinpath(path)
        self.metadata = pd.read_csv(Path.cwd().joinpath(metadata_path))
        self.names = self.encode_names()
        self.species = self.encode_species()
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

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
        samples = []
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
                sample = [np.array(mel_spec).reshape((mel_spec.shape[0], mel_spec.shape[1], 1)), np.array(self.names[idx]), np.array(self.species[idx])]
                start += int(self.max_len * hz // (1 - self.overlapping))
                samples.append(sample)

        for i in range(len(samples)):
            if self.transform is not None:
                samples[i] = self.transform(samples[i])
        print("Data Loaded !")

        return samples

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1], self.data[idx][2]
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

# ds = AudioDataset(transform=ToTensor() , max_len=3, overlapping=0, noise_reduction=True)
# print(len(ds))