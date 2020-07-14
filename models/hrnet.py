import torch

from torch import nn


class Conv(nn.Module):
	def __init__(self, ch, kernel_size=3, padding=1):
		super(Conv, self).__init__()
		self.conv = nn.Conv2d(ch, ch, kernel_size, padding=padding)
		self.bn = nn.BatchNorm2d(ch)
		self.relu = nn.ReLU()
	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)
		return x


class ConvBlock(nn.Module):
	def __init__(self, ch, resnet=True):
		super(ConvBlock, self).__init__()
		self.conv1 = Conv(ch)
		self.conv2 = Conv(ch)
		self.conv = nn.Conv2d(ch, ch, 3, padding=1)
		self.bn = nn.BatchNorm2d(ch)
		self.relu = nn.ReLU()
		self.resnet = resnet
	def forward(self, x):
		x = self.conv1(x)
		identity = x
		x = self.conv2(x)
		x = self.conv(x)
		x = self.bn(x)
		if self.resnet:
			x += identity
		x = self.relu(x)
		return x


class UpSampling(nn.Module):
	def __init__(self, ch, up_factor):
		super(UpSampling, self).__init__()
		self.up = nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=True)
		self.conv = nn.Conv2d(ch * up_factor, ch, 1)
	def forward(self, x):
		x = self.up(x)
		x = self.conv(x)
		return x


class DownSampling(nn.Module):
	def __init__(self, ch, down_factor):
		super(DownSampling, self).__init__()
		self.down = nn.Conv2d(ch // down_factor, ch, 3, down_factor, 1)
	def forward(self, x):
		x = self.down(x)
		return x


class HRBlock(nn.Module):
	def __init__(self, ch, index, num_conv_block_per_list=5):
		super(HRBlock, self).__init__()
		self.index = index
		self.num_conv_block_per_list = num_conv_block_per_list

		self.parallel_conv_lists = nn.ModuleList()
		for i in range(index):
			ch_i = ch * 2**i
			conv_list = nn.ModuleList()
			for j in range(num_conv_block_per_list):
				conv_list.append(ConvBlock(ch_i, ch_i))
			self.parallel_conv_lists.append(conv_list)

		self.up_conv_lists = nn.ModuleList()
		for i in range(index - 1):
			ch_i = ch * 2**i
			conv_list = nn.ModuleList()
			for j in range(i + 1, index):
				up_factor = 2 ** (j-i)
				conv_list.append(UpSampling(ch_i, up_factor))
			self.up_conv_lists.append(conv_list)

		self.down_conv_lists = nn.ModuleList()
		for i in range(1, index + 1):
			ch_i = ch * 2**i
			conv_list = nn.ModuleList()
			for j in range(i):
				down_factor = 2 ** (i - j)
				conv_list.append(DownSampling(ch_i, down_factor))
			self.down_conv_lists.append(conv_list)

	def forward(self, x_list):
		parallel_res_list = []
		for i in range(self.index):
			x = x_list[i]
			for j in range(self.num_conv_block_per_list - 1):
				x = self.parallel_conv_lists[i][j](x)
				if j == self.num_conv_block_per_list - 2:
					parallel_res_list.append(x)
		final_res_list = []
		for i in range(self.index + 1):
			if i == self.index:
				x = torch.stack([m(t) for t, m in zip(parallel_res_list, self.down_conv_lists[-1])])
				x = torch.sum(x, dim=0)
			else:
				x = parallel_res_list[i]
				x = self.parallel_conv_lists[i][-1](x)
				if i != self.index - 1:
					res_list = parallel_res_list[i+1:]
					up_x = torch.stack([m(t) for t, m in zip(res_list, self.up_conv_lists[i])])
					up_x = torch.sum(up_x, dim=0)
					x += up_x
				if i != 0:
					res_list = parallel_res_list[:i]
					down_x = torch.stack([m(t) for t, m in zip(res_list, self.down_conv_lists[i - 1])])
					down_x = torch.sum(down_x, dim=0)
					x += down_x
			final_res_list.append(x)
		return final_res_list


class HRNet(nn.Module):
	'''
	Deep High-Resolution Representation Learning for Visual Recognition
	https://arxiv.org/pdf/1908.07919.pdf
	2020
	'''
	def __init__(self, in_ch, mid_ch, out_ch, num_stage=4, regressive=True):
		super(HRNet, self).__init__()
		self.init_conv = nn.Conv2d(in_ch, mid_ch, 1)
		self.last_conv = nn.Conv2d(mid_ch * (num_stage + 1), out_ch, 1)
		self.num_stage = num_stage
		self.regressive = regressive
		self.hr_blocks = nn.ModuleList()
		for i in range(num_stage):
			self.hr_blocks.append(HRBlock(mid_ch, i + 1))
		self.up_convs = nn.ModuleList()
		for i in range(num_stage):
			self.up_convs.append(UpSampling(mid_ch, 2 ** (i + 1)))

	def forward(self, x):
		x = self.init_conv(x)
		x_list = [x]
		for i in range(self.num_stage):
			x_list = self.hr_blocks[i](x_list)

		res_list = [x_list[0]]
		for t, m in zip(x_list[1:], self.up_convs):
			res_list.append(m(t))

		x = torch.cat(res_list, dim=1)
		x = self.last_conv(x)
		return x if self.regressive else torch.sigmoid(x).clamp(1e-4, 1 - 1e-4)

