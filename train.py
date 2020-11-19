import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image
import argparse
from net_model import *
from dataset import *
import time

class trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_net = D_Net().to(self.device)
        self.g_net = G_Net().to(self.device)

    def train(self):
        batch_size = 100
        num_epoch = 200

        if not os.path.exists("./cartoon_img"):
            os.mkdir("./cartoon_img")
        if not os.path.exists("./cartoon_params"):
            os.mkdir("./cartoon_params")

        cartoon = Mydata(r".\faces")
        loader = DataLoader(cartoon, batch_size=batch_size, shuffle=True, num_workers=4)

        d_weight_file = r"cartoon_params/d_net.pth"
        g_weight_file = r"cartoon_params/g_net.pth"

        if not os.path.exists("cartoon_params"):
            os.mkdir("cartoon_params")
        if os.path.exists(d_weight_file):
            self.d_net.load_state_dict(torch.load(d_weight_file))
        if os.path.exists(g_weight_file):
            self.g_net.load_state_dict(torch.load(g_weight_file))

        loss_fn = nn.BCELoss()
        d_opt = torch.optim.Adam(self.d_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
        g_opt = torch.optim.Adam(self.g_net.parameters(), lr=0.0002, betas=(0.5, 0.999))

        for epoch in range(num_epoch):
            x = time.time()
            for i, (image) in enumerate(loader):
                N = image.size(0)
                real_img = image.to(self.device)
                real_label = torch.ones(N, 1, 1, 1).to(self.device)
                # 判别器
                fake_label = torch.zeros(N, 1, 1, 1).to(self.device)
                real_out = self.d_net(real_img)
                d_real_loss = loss_fn(real_out, real_label)
                z = torch.randn(N, 100, 1, 1).to(self.device)
                fake_img = self.g_net(z)
                fake_out = self.d_net(fake_img)
                d_fake_loss = loss_fn(fake_out, fake_label)
                d_loss = d_real_loss + d_fake_loss

                d_opt.zero_grad()
                d_loss.backward()
                d_opt.step()
                # 生成器
                if i % 5 == 0:
                    z = torch.randn(N, 100, 1, 1).to(self.device)
                    fake_img = self.g_net(z)
                    fake_out = self.d_net(fake_img)
                    g_loss = loss_fn(fake_out, real_label)
                    g_opt.zero_grad()
                    g_loss.backward()
                    g_opt.step()

                if i % 200 == 0:
                    print("Epoch:{}/{},d_loss:{:.3f},g_loss:{:.3f},"
                          "d_real:{:.3f},d_fake:{:.3f}".
                          format(epoch, num_epoch, d_loss.item(), g_loss.item(),
                                 real_out.data.mean(), fake_out.data.mean()))

                    real_image = real_img.cpu().data
                    save_image(real_image, "./cartoon_img/epoch{}-iteration{}-real_img.jpg".
                               format(110+epoch,i), nrow=10, normalize=True, scale_each=True)
                    fake_image = fake_img.cpu().data
                    save_image(fake_image, "./cartoon_img/epoch{}-iteration{}-fake_img.jpg".
                               format(110+epoch,i), nrow=10, normalize=True, scale_each=True)
                    torch.save(self.d_net.state_dict(), d_weight_file)
                    torch.save(self.g_net.state_dict(), g_weight_file)
                    y = time.time()

if __name__ == '__main__':

    t = trainer()
    t.train()