import torch
import utils
import time
import os
import pickle
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from dataloader import dataloader0, dataloader1, dataloader2, dataloader3
import torchvision.utils as vutils
from torch.autograd import Variable


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        out = residual + x
        out = self.relu(out)

        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)

        return out


class Generator(nn.Module):
    def __init__(self, in_channels):
        super(Generator, self).__init__()
        self.in_channels = in_channels

        self.encoder = nn.Sequential(
            ConvLayer(self.in_channels, 32, 9, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            ConvLayer(32, 64, 3, 2),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            ConvLayer(64, 128, 3, 2),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 9, 1, 4),
            #nn.InstanceNorm2d(3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        out = self.decoder(x)

        return out


def VGG16_path4():
    conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    return conv


def VGG16_path3():
    conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=False),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    return conv


def VGG16_path2():
    conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=False),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=False),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    return conv


def VGG16_path1():
    conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=False),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=False),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=False),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    return conv


def Path1(x):
    model = VGG16_path1()
    vgg16 = models.vgg16(pretrained=True)
    pretrained_dict = vgg16.state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    if torch.cuda.is_available():
        model = model.cuda()
    out = model(x)

    return out


def Path2(x):
    model = VGG16_path2()
    vgg16 = models.vgg16(pretrained=True)
    pretrained_dict = vgg16.state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    if torch.cuda.is_available():
        model = model.cuda()
    out = model(x)

    return out


def Path3(x):
    model = VGG16_path3()
    vgg16 = models.vgg16(pretrained=True)
    pretrained_dict = vgg16.state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    if torch.cuda.is_available():
        model = model.cuda()
    out = model(x)

    return out


def Path4(x):
    model = VGG16_path4()
    vgg16 = models.vgg16(pretrained=True)
    pretrained_dict = vgg16.state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    if torch.cuda.is_available():
        model = model.cuda()
    out = model(x)

    return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 1, 4, 2, 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 1, 4, 2, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 1, 4, 2, 1)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 1, 4, 2, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(1 * 3 * 12, 1)
        )

    def forward(self, path1, path2, path3, path4):
        out1 = self.conv1(path1)
        out2 = self.conv2(path2)
        out3 = self.conv3(path3)
        out4 = self.conv4(path4)
        out = torch.cat((out1, out2, out3, out4), 1)
        out = out.view(-1, 1 * 3 * 12)
        out = self.fc(out)

        return out


class GAN(object):
    def __init__(self, args):
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.gpu_mode = args.gpu_mode
        self.input_size = args.input_size

        self.dataloader0 = dataloader0(input_size=self.input_size, batch_size=self.batch_size)
        self.dataloader1 = dataloader1(input_size=self.input_size, batch_size=self.batch_size)
        self.dataloader2 = dataloader2(input_size=self.input_size, batch_size=self.batch_size)
        self.dataloader3 = dataloader3(input_size=self.input_size, batch_size=self.batch_size)

        data = self.dataloader0.__iter__().__next__()[0]

        self.G = Generator(in_channels=3)
        self.D = Discriminator()
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G = self.G.cuda()
            self.D = self.D.cuda()
            self.MSELoss = nn.MSELoss().cuda()
        else:
            self.MSELoss = nn.MSELoss()

        print('------------ Networks Architecture ------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        self.sample_z = torch.rand(self.batch_size, 100, 1, 1)
        if self.gpu_mode:
            self.sample_z = self.sample_z.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real = torch.ones((self.batch_size, 1))
        self.y_fake = torch.zeros((self.batch_size, 1))
        if self.gpu_mode:
            self.y_real = self.y_real.cuda()
            self.y_fake = self.y_fake.cuda()

        self.D.train()
        print('Training start!')
        start_time = time.time()

        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()

            for iter, (x, y) in enumerate(self.dataloader1):
                if iter == self.dataloader1.dataset.__len__() // self.batch_size:
                    break

                if self.gpu_mode:
                    x = x.cuda()

                self.D_optimizer.zero_grad()
                path4 = Path4(x)
                path3 = Path3(x)
                path2 = Path2(x)
                path1 = Path1(x)
                D_real = self.D(path1=path1, path2=path2, path3=path3, path4=path4)
                D_real_loss = self.MSELoss(D_real, self.y_real)

                D_loss = D_real_loss
                D_loss.backward()
                self.D_optimizer.step()
                if ((iter + 1) % 100) == 0:
                    print('Epoch: [%2d] [%4d/%4d] D_loss: %.8f' %
                          (
                              (epoch + 1), (iter + 1), self.dataloader0.dataset.__len__() // self.batch_size,
                              D_loss.item()
                          ))

            for iter, (x, y) in enumerate(self.dataloader0):
                if iter == self.dataloader0.dataset.__len__() // self.batch_size:
                    break

                if self.gpu_mode:
                    x = x.cuda()

                if epoch == 0 and iter == 0:
                    # noise data is given
                    fixed_noise = x[:8].repeat(10, 1, 1, 1)
                    global fixed_img_v
                    fixed_img_v = Variable(fixed_noise)

                    pickle.dump(fixed_noise, open("fixed_noise.p", "wb"))

                    if self.gpu_mode:
                        fixed_img_v = fixed_img_v.cuda()

                self.D_optimizer.zero_grad()
                
                path41 = Path4(x)
                path31 = Path3(x)
                path21 = Path2(x)
                path11 = Path1(x)
                D_fake_1 = self.D(path1=path11, path2=path21, path3=path31, path4=path41)
                D_fake_1_loss = self.MSELoss(D_fake_1, self.y_fake)

                G = self.G(x)
               
                path42 = Path4(G)
                path32 = Path3(G)
                path22 = Path2(G)
                path12 = Path1(G)
                D_fake_2 = self.D(path1=path12, path2=path22, path3=path32, path4=path42)
                D_fake_2_loss = self.MSELoss(D_fake_2, self.y_fake)

                D_loss = D_fake_1_loss + D_fake_2_loss
                self.train_hist['D_loss'].append(D_loss)
                D_loss.backward()
                self.D_optimizer.step()

                self.G_optimizer.zero_grad()
                G_ = self.G(x)
                
                path43 = Path4(G_)
                path33 = Path3(G_)
                path23 = Path2(G_)
                path13 = Path1(G_)
                D_fake_3 = self.D(path1=path13, path2=path23, path3=path33, path4=path43)

                G_loss_1 = self.MSELoss(D_fake_3, self.y_real)
                G_loss_2 = self.MSELoss(G_, x)

                G_loss = 750 * G_loss_1 + 0.2 * G_loss_2
                self.train_hist['G_loss'].append(G_loss)
                G_loss.backward()
                self.G_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print('Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f' %
                          (
                              (epoch + 1), (iter + 1), self.dataloader0.dataset.__len__() // self.batch_size,
                              D_loss.item(), G_loss.item()
                          ))
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

            if not os.path.exists(self.result_dir + '/' + 'COCO'):
                os.makedirs(self.result_dir + '/' + 'COCO')
            root = os.path.join(self.result_dir, 'COCO')
            #with torch.no_grad():
                #self.visualize_results((epoch + 1))
            img_fake = self.G(fixed_img_v)
            vutils.save_image(img_fake, '%s/reconst_epoch%03d.png' % (root, epoch+1), normalize=True)

        self.train_hist['total_time'].append(time.time() - start_time)
        print('Average epoch time: %.2f, total %d epochs time: %.2f' % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print('Training finished. The results saved')

        self.save()

        # utils.generate_animation(
        #     self.result_dir + '/' + self.dataset + '/' + self.model_name + '2' + '/' + self.model_name, self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, 'Loss_plot'), 'Pyramid_GAN')

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + 'COCO'):
            os.makedirs(self.result_dir + '/' + 'COCO')

        total_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(total_num_samples)))

        if fix:
            samples = self.G(self.sample_z)
        else:
            sample_z = torch.rand((self.batch_size, 100))
            if self.gpu_mode:
                sample_z = sample_z.cuda()
            samples = self.G(sample_z)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + 'COCO' + '/' + 'Pyramid_GAN' + 'epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, 'Model_dict')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.D.parameters(), os.path.join(save_dir, 'Pyramid_GAN' + '_D.pkl'))
        torch.save(self.G.parameters(), os.path.join(save_dir, 'Pyramid_GAN' + '_G.pkl'))

        with open(os.path.join(save_dir, 'Pyramid_GAN' + '_history.pkl', 'wb')) as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, 'Model_dict')

        self.D.load_state_dict(torch.load(os.path.join(save_dir, 'Pyramid_GAN' + '_D.pkl')))
        self.G.load_state_dict(torch.load(os.path.join(save_dir, 'Pyramid_GAN' + '_G.pkl')))

