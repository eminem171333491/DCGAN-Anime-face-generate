from torch.utils import data
import os
from PIL import Image
from torchvision import transforms

tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229, 0.224, 0.225])
        ])

class Mydata(data.Dataset):
    def __init__(self,data_path):
        self.dataset = []
        for im in os.listdir(data_path):
            self.dataset.append(os.path.join(data_path,im))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        x = Image.open(data).convert("RGB")
        img = tf(x)
        return img

if __name__ == '__main__':
    data_path = r".\Cartoon_faces\faces"
    mydata = Mydata(data_path)
    data = data.DataLoader(mydata,10,shuffle=True)
    for images in data:
        # print(x.shape)
        print(images.size(0))