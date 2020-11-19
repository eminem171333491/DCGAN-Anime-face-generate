import torch.nn as nn



class D_Net(nn.Module):
    def __init__(self):
        super(D_Net,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,5,3,1,bias=False),
            nn.LeakyReLU(0.2,True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256,512,4,2,1,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512,1,4,1,0,bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class G_Net(nn.Module):
    def __init__(self):
        super(G_Net,self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(100,512,4,1,0,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1,bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(64,3,5,3,1,1,bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x







