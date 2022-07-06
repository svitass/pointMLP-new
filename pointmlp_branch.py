import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# MLP + BN + ReLU
# 当kernel_size=1时，Conv1d实现了MLP
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Block,self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias),  # (batch,out_channels,length)
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        """
        :param x: (batch, in_channel, length)
        :return:
        """
        x = self.net(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels, res_expansion=1.0, bias=True):
        super(ResBlock,self).__init__()
        self.net = nn.Sequential(   # MLP + BN + ReLU + MLP + BN
            nn.Conv1d(in_channels=channels, out_channels=int(channels * res_expansion), kernel_size=1, bias=bias),
            nn.BatchNorm1d(int(channels * res_expansion)),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=int(channels * res_expansion), out_channels=channels, kernel_size=1, bias=bias),
            nn.BatchNorm1d(channels)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: (batch, channels, length)
        :return:
        """
        x = x + self.net(x)
        x = self.act(x)
        return x


"""
用于学习local region间的共享权重
"""
class PreExtraction(nn.Module):
    def __init__(self,in_channels, out_channels, bias=True):
        super(PreExtraction,self).__init__()
        self.operation = nn.Sequential(
            Block(in_channels, out_channels, bias=bias),
            ResBlock(out_channels, res_expansion=1.0, bias=bias),
            ResBlock(out_channels, res_expansion=1.0, bias=bias),  # (batch*fps_num ,feature_dim, k)
            nn.AdaptiveMaxPool1d(1)  # (batch*fps_num ,feature_dim, 1)
        )

    def forward(self, x):  # (batch, fps_num, k, feature_dim)
        # 1. 先将x转换为conv1d的输入格式 (batch, channel, length)
        b,n,k,d = x.size()
        x = x.permute(0,1,3,2)  # (batch, fps_num, feature_dim, k)
        x = x.reshape(-1,d,k)
        # 2. 喂入Conv1d
        x = self.operation(x) # (batch*fps_num ,feature_dim, 1)
        x = x.reshape(b,n,-1) # (batch, fps_num, feature_dim)
        x = x.permute(0,2,1) # (batch, feature_Dim, fps_num)  便于输入PosExtraction
        return x


"""
提取深度特征
"""
class PosExtraction(nn.Module):
    def __init__(self,channels,bias=True):
        super(PosExtraction,self).__init__()
        self.operation = nn.Sequential(
            ResBlock(channels, res_expansion=1.0, bias=bias),
            ResBlock(channels, res_expansion=1.0, bias=bias)
        )

    def forward(self, x):
        x = self.operation(x)
        return x


"""
FPS原理：（使采样点之间尽可能远离）
1. 输入点云有N个点，从点云中选取一个点P0作为起始点，得到采样点集合S={P0}
2. 计算所有点到P0的距离，构成N维数组L，从中选择最大值对应的点作为P1, 更新采样点集合S={P0,P1}
3. 计算所有点到P1的距离，对于每一个点Pi,其距离P1的距离如果小于L[i], 则更新L[i]=d(Pi,P1),
因此，数组L中存储的一直是每一个点到采样点集合S的最近距离
4. 选取L中最大值对应的点作为P2, 更新采样点集合S={P0,P1,P2}
5. 重复2-4步，一直采样到N个目标采样点为止
"""
def furthest_point_sample(points, num):
    """
    :param points: 点云的xyz (batch_size, points_num, 3)
    :param num: FPS采样的点数
    :return:
    """
    device = points.device
    batch, points_num,_ = points.shape
    # (batch, points_num) 用来存储采样点的index
    sample_points_ids = torch.zeros(batch,num,dtype=torch.long).to(device)
    min_dists = torch.ones(batch,points_num).to(device) * 1e10   # 用于保存每个点到集合S的最近距离
    farthest_points = torch.randint(0, points_num, (batch,), dtype=torch.long).to(device) # 初始化一个最远点
    batch_ids = torch.arange(batch, dtype=torch.long).to(device)
    for i in range(num):  # 最终生成num个采样点  缺点：串行，耗时   能否空间换时间？？？
        sample_points_ids[:,i] = farthest_points
        xyz = points[batch_ids,farthest_points,:].view(batch,1,3)  # 获取每个最远点的xyz坐标
        dists = torch.sum((points - xyz)**2, -1)  # 计算所有点到当前采样点的距离
        min_dists = torch.min(min_dists, dists)
        farthest_points = torch.max(min_dists, -1)[1]  # 最大距离对应的点索引
    return sample_points_ids


"""
为每个sample_point查找k个邻居点: 
1. 计算所有点到sample_point的距离
2. 选择距离sample_point最近的k个邻居点
3. 根据距离，为每个邻居点计算权值，用来说明邻居点特征对采样点的影响性（距离和权值成反比）
(1) d<=sigma1, w=1
(2) sigma1<d<sigma2, w递减
(3) d>=sigma3, w=0
"""
def knn_point(points,sample_points,k):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    :param points:
    :param sample_points:
    :param k:
    """
    # 1. 计算距离
    batch,ori_num,_ = points.shape  # (2,1024,3)
    _,sample_num,_ = sample_points.shape # (2,512,3)
    dist1 = torch.sum(points ** 2, -1)  # (2,1024)
    dist1 = dist1.view(batch,1, ori_num)  # (2,1,1024)
    dist2 = torch.sum(sample_points ** 2, -1) # (2,512)
    dist2 = dist2.view(batch, sample_num,1)  # (2,512,1)
    dist3 = torch.matmul(sample_points,points.permute(0,2,1))   # (2,512,1024)
    dist = dist1 + dist2 -2 * dist3
    # 2. 选择前k个最小的值
    _, group_idx = torch.topk(dist, k, dim=-1, largest=False, sorted=False) # (2,512,24) 24表示point index
    # 3. 计算权重
    batch_idx = torch.arange(batch, dtype=torch.long).view(batch, 1, 1).repeat(1,sample_num,k)
    sample_idx = torch.arange(sample_num,dtype=torch.long).view(1,sample_num,1).repeat(batch,1,k)
    weights= torch.exp(-dist[batch_idx,sample_idx,group_idx])  # (2,512,24)
    return group_idx,weights


"""
1. FPS从点云中采样
2. 为每个采样点选24个邻居点(knn)
3. normalize邻居点的特征 使得采样点的特征=采样点特征+邻居点的特征
"""
class GeometricAffine(nn.Module):
    def __init__(self, channels, fps_num, kneighbors):
        """
        :param channels:
        :param fps_num: FPS采样点的个数
        :param kneighbors: KNN聚类的点数，即每个采样点周围有多少个邻居点
        """
        super(GeometricAffine,self).__init__()
        self.fps_num = fps_num
        self.kneighbors = kneighbors
        # alpha, beta:  R^d, 可学习的参数
        self.alpha = nn.Parameter(torch.ones([1,1,1,channels]))
        self.beta = nn.Parameter(torch.zeros([1,1,1,channels]))

    def forward(self, points, points_features, fps_idx=None):
        """
        :param points: 点的xyz信息 （batch_size, points_num, 3）
        :param points_features: 点的特征 (batch_size, points_num, features_dim)
        :return:  返回采样点及采样点的特征
        """
        # 1. FPS采样, 获取采样点的xyz坐标和特征
        if fps_idx == None:
            fps_idx = furthest_point_sample(points, self.fps_num) # (batch, fps_num)
        batch,_ = fps_idx.shape
        batch_idx = torch.arange(batch,dtype=torch.long).view(batch,1).repeat(1,self.fps_num)
        sample_points = points[batch_idx, fps_idx, :]   # (batch, fps_num, 3)
        sample_points_features = points_features[batch_idx, fps_idx, :]   # (batch, fps_num, feature_dim)

        # 2. KNN选取邻居点, 获取邻居点的xyz坐标和特征
        group_idx,weights = knn_point(points, sample_points, self.kneighbors) # (batch, fps_num, kneighbors)
        batch_idx = torch.arange(batch,dtype=torch.long).view(batch,1,1).repeat(1,self.fps_num,self.kneighbors)
        group_points = points[batch_idx, group_idx, :]  # (batch, fps_num, kneighbors, 3)  (2, 512, 24, 3)
        group_features = points_features[batch_idx, group_idx, :]  # (batch, fps_num, kneighbors, feature_dim)

        # 3. normalize邻居点的特征
        mean = sample_points_features.unsqueeze(dim=-2)  # (batch, fps_num, 1, feature_dim)
        std = torch.std((group_features-mean).reshape(batch,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1) # (batch, 1, 1, 1)
        group_features = (group_features-mean)/(std + 1e-5)
        group_features = self.alpha*group_features+self.beta  # (batch, fps_num, kneighbors, feature_dim)


        # 4. 邻居点特征加权
        feature_dim = points_features.shape[-1]
        # weights = weights.unsqueeze(dim=-1).repeat(1,1,1,feature_dim)
        # group_features = weights.mul(group_features)

        # 5. concate采样点和邻居点的特征
        sample_points_features = sample_points_features.view(batch,self.fps_num,1,feature_dim).repeat(1,1,self.kneighbors,1)
        sample_points_features = torch.cat([group_features,sample_points_features], dim=-1)
        return fps_idx,sample_points,sample_points_features




class PointMLP(nn.Module):
    def __init__(self, points_num=1024, class_num=15):
        """
        :param points_num: object中点的个数
        :param class_num: 类别数
        """
        super(PointMLP,self).__init__()
        self.stages = 4  # 全局划分为4个stage
        self.class_num = class_num
        k_neighbors = 24  # knn聚类时邻居点的个数
        self.embedding = Block(3, 64, bias=True)  # 用于计算所有点的特征
        # 为每个state添加特征提取模块
        self.local_grouper_list = nn.ModuleList([
            GeometricAffine(64, 512, k_neighbors),  # feature_dim:64 fps_num:512
            GeometricAffine(128,256, k_neighbors), # GeometricAffine最后concate了采样点特征和邻居点特征，feature_dim * 2
            GeometricAffine(256,128,k_neighbors),
            GeometricAffine(512,64,k_neighbors)
        ])
        self.pre_blocks_list = nn.ModuleList([
            PreExtraction(128, 128, bias=True),
            PreExtraction(256, 256, bias=True),
            PreExtraction(512, 512, bias=True),
            PreExtraction(1024,1024,bias=True)
        ])
        self.pos_blocks_list = nn.ModuleList([
            PosExtraction(128,bias=True),
            PosExtraction(256,bias=True),
            PosExtraction(512,bias=True),
            PosExtraction(1024,bias=True)
        ])
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.class_num)
        )


    def forward(self, x):
        """
        :param x: 点云的xyz坐标(batch_size, points_num, 3)
        :return:
        """
        # 1. 把x转为Conv1d的输入（batch,channel,length），然后计算所有点的特征
        points = x
        device = x.device
        batch, points_num, _ = points.shape
        features = x.permute(0,2,1)  # (batch, 3, points_num)
        features = self.embedding(features)  # (batch, 64, points_num)
        # for i in range(self.stages):
        #     features = features.permute(0,2,1)  # (batch, points_num, 64)
        #     points,features = self.local_grouper_list[i](points,features)
        #     features = self.pre_blocks_list[i](features) # (batch, feature_dim, fps_num)
        #     features = self.pos_blocks_list[i](features) # (batch, feature_dim, fps_num)
        # features = F.adaptive_max_pool1d(features,1)  # (batch, 1024, 1)
        # 调整features维度，是其作为分类器的输入
        # features = features.squeeze(dim=-1)

        ## 2. 实验， 尝试多分支提取特征，作为损失项
        idx = torch.zeros(batch,points_num,dtype=torch.long).to(device)
        batch_idx = torch.arange(0, batch, dtype=torch.long).to(device)
        idx[batch_idx] = torch.arange(0, points_num, dtype=torch.long).to(device)

        # 2.1 原始分支
        features_branch1 = features.permute(0,2,1)   # (batch, points_num, 64)
        fps_idx_branch1,points_branch1,features_branch1 = self.local_grouper_list[0](points,features_branch1)
        features_branch1 = self.pre_blocks_list[0](features_branch1) # (batch, feature_dim, fps_num)
        features_branch1 = self.pos_blocks_list[0](features_branch1)  # (batch, feature_dim, fps_num)

        # 2.2 loss分支
        fps_idx_branch2 = torch.zeros(batch,points_num-512,dtype=torch.long).to(device)
        for i in range(batch):
            fps_idx_branch2[i,:] = idx[i][~torch.isin(idx[i],fps_idx_branch1[i])]
        features_branch2 = features.permute(0,2,1)
        _,points_branch2,features_branch2 = self.local_grouper_list[0](points,features_branch2,fps_idx_branch2)
        features_branch2 = self.pre_blocks_list[0](features_branch2)
        features_branch2 = self.pos_blocks_list[0](features_branch2)

        for i in range(1, self.stages):
            features_branch1 = features_branch1.permute(0,2,1)
            fps_idx_branch1, points_branch1, features_branch1 = self.local_grouper_list[i](points_branch1, features_branch1)
            features_branch1 = self.pre_blocks_list[i](features_branch1)  # (batch, feature_dim, fps_num)
            features_branch1 = self.pos_blocks_list[i](features_branch1)  # (batch, feature_dim, fps_num)

            features_branch2 = features_branch2.permute(0, 2, 1)
            fps_idx_branch2, points_branch2, features_branch2 = self.local_grouper_list[i](points_branch2,
                                                                                           features_branch2)
            features_branch2 = self.pre_blocks_list[i](features_branch2)  # (batch, feature_dim, fps_num)
            features_branch2 = self.pos_blocks_list[i](features_branch2)  # (batch, feature_dim, fps_num)

        features_branch1 = F.adaptive_max_pool1d(features_branch1,1)  # (batch, 1024, 1)
        features_branch1 = features_branch1.squeeze(dim=-1)  # (batch, 1024)

        features_branch2 = F.adaptive_max_pool1d(features_branch2, 1)
        features_branch2 = features_branch2.squeeze(dim=-1)

        features = self.classifier(features_branch1)  # (batch, class_num)
        return features,features_branch1,features_branch2
        # return features

if __name__ == '__main__':
    data = torch.rand(2,1024,3)
    model = PointMLP(points_num=1024, class_num=15)
    out,_,_ = model(data)
    print("features:",out)
    print("out.shape:",out.shape)



