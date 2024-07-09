import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, training=True):
        super(UNet, self).__init__()
        self.training = training
        self.sigmoid = nn.Sigmoid()
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2 = nn.Conv3d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3 = nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4 = nn.Conv3d(128, 256, 3, stride=1, padding=1)
        self.encoder5 = nn.Conv3d(256, 512, 3, stride=1, padding=1)

        self.decoder1 = nn.Conv3d(512, 256, 3, stride=1, padding=1)  # b, 16, 5, 5
        self.decoder2 = nn.Conv3d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 = nn.Conv3d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(32, 1, 3, stride=1, padding=1)

        self.map4 = nn.Sequential(
            #nn.Conv3d(2, out_channel, 1, 1),
            #nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear'),
            # nn.Softmax(dim =1)
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear'),
            # nn.Softmax(dim =1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(16, 16, 16), mode='trilinear'),
            # nn.Softmax(dim =1)
        )

    def forward(self, x):
        a = self.encoder1(x)
        out = F.relu(F.max_pool3d(self.encoder1(x), 2, 2))
        t1 = out
        out = F.relu(F.max_pool3d(self.encoder2(out), 2, 2))
        t2 = out
        out = F.relu(F.max_pool3d(self.encoder3(out), 2, 2))
        t3 = out
        out = F.relu(F.max_pool3d(self.encoder4(out), 2, 2))
        t4 = out
        out0 = F.relu(F.max_pool3d(self.encoder5(out), 2, 2))

        # t2 = out
        # out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2,2),mode ='trilinear'))
        # print(out.shape,t4.shape)
        b = self.decoder1(out0)
        out1 = F.relu(F.interpolate(self.decoder1(out0), scale_factor=(2, 2, 2), mode='trilinear'))

        out2 = torch.add(out1, t4)
        output1 = self.map1(out2)
        out3 = F.relu(F.interpolate(self.decoder2(out2), scale_factor=(2, 2, 2), mode='trilinear'))
        out4 = torch.add(out3, t3)
        output2 = self.map2(out4)
        out5 = F.relu(F.interpolate(self.decoder3(out4), scale_factor=(2, 2, 2), mode='trilinear'))
        out6 = torch.add(out5, t2)
        output3 = self.map3(out6)
        out7 = F.relu(F.interpolate(self.decoder4(out6), scale_factor=(2, 2, 2), mode='trilinear'))
        out8 = torch.add(out7, t1)

        out9 = F.relu(F.interpolate(self.decoder5(out8), scale_factor=(2, 2, 2), mode='trilinear'))

        # out9 = self.map4(out9)

        # out9 = torch.argmax(out9, dim=1)


        # out9 = self.sigmoid(out9)
        #generated_image = out9
        # print(out.shape)
        # print(output1.shape,output2.shape,output3.shape,output4.shape)

        # # 查看第一个样本，第一个通道，第60个切片
        # slice_60_output1 = output1[0, 0, 59, :, :]
        # slice_60_output2 = output2[0, 0, 59, :, :]
        # slice_60_output3 = output3[0, 0, 59, :, :]
        # slice_60_out9 = out9[0, 0,59, :, :]
        #
        # # 将张量从GPU移到CPU，并转换为NumPy数组
        # slice_60_output1 = slice_60_output1.cpu().detach().numpy()
        # slice_60_output2 = slice_60_output2.cpu().detach().numpy()
        # slice_60_output3 = slice_60_output3.cpu().detach().numpy()
        # slice_60_out9 = slice_60_out9.cpu().detach().numpy()
        #
        # # 使用 matplotlib 查看这个切片
        # import matplotlib.pyplot as plt
        #
        # plt.imshow(slice_60_output1, cmap='gray')
        # plt.title('slice_60_output1')
        # plt.show()
        # plt.imshow(slice_60_output2, cmap='gray')
        # plt.title('slice_60_output2')
        # plt.show()
        # plt.imshow(slice_60_output3, cmap='gray')
        # plt.title('slice_60_output3')
        # plt.show()
        # plt.imshow(slice_60_out9, cmap='gray')
        # plt.title('slice_60_out9')
        # plt.show()


        if self.training is True:
            return output1, output2, output3, out9
        else:
            return out9
        # return generated_image