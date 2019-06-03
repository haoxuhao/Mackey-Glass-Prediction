from torch.utils.data import Dataset,DataLoader
import random
import numpy as np 

class MG_Dataset(Dataset):
    def __init__(self, data, input_dim=5, data_range=[201, 3200], train=True, is_shuffle=True):
        super(MG_Dataset, self).__init__()
        self.data = data
        self.train = train
        self.data_range = data_range
        self.samples = [id for id in range(data_range[0], data_range[1])]

        # if is_shuffle and train:
        #     random.shuffle(self.samples)

        self.D = input_dim
        self.P = 2
        self.F = 25
        self.nSamples = len(self.samples)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index < self.nSamples, 'index range error' 
        t = self.samples[index]

        features = []
        for i in range(self.D):
            features.append(self.data[t - self.P*i])
        target = self.data[t+self.F]

        return np.array(features), target

def get_dataloader(data, train=False, batch_size=64, input_dim=5):
    # print("dataset shape",data.shape)
    if train:
        data_range=[201, 3200]
        loader = DataLoader(
            MG_Dataset(data, data_range=data_range, is_shuffle=True, input_dim=input_dim),
            batch_size=batch_size
        )
    else:
        data_range = [5001, 5500]
        loader = DataLoader(
            MG_Dataset(data, data_range=data_range, is_shuffle=False, train=False, input_dim=input_dim),
            batch_size=batch_size
        )
    return loader



if __name__ == "__main__":
    data=np.load("./data/data.npy")
    train_loader = get_dataloader(data, train=True)
    for i,(feature, target) in enumerate(train_loader):
        print(feature.shape)
        print(target.shape)