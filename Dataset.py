import torch


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_num, transform=None):
        self.transform = transform
        self.data_num = data_num
        self.data = []
        self.label = []
        for x in range(self.data_num):
            self.data.append(x)  # 0 から (data_num-1) までのリスト
            self.label.append(x % 2 == 0)  # 偶数ならTrue 奇数ならFalse

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label
