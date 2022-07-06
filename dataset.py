from torch.utils.data import Dataset
import h5py
import numpy as np
import vedo

def load_scanobjectnn_data(mode):
    """
    加载ScanObjectNN数据
    :param mode: ['train', 'test']
    :return: 返回所有的data和label
    """
    if mode=="train":   # 加载训练数据集
        path = 'data/ScanObjectNN/training_objectdataset_augmentedrot_scale75.h5'
    else:  # 加载测试数据集
        path = 'data/ScanObjectNN/test_objectdataset_augmentedrot_scale75.h5'
    f = h5py.File(path, mode="r")  # 读取hdf5文件, hdf5以group和dataset的形式存储数据
    # 读取数据集  train:(11416,2048,3)  test: (2882,2048,3)
    data = f['data'][:].astype('float32')
    # 读取标签  train: (11416,)  test: (2882,)
    label = f['label'][:].astype('int64')
    f.close()
    return data,label

def load_modelnet40_data(mode):
    if mode=="train":
        pathList = [
            'data/ModelNet40/ply_data_train0.h5',
            'data/ModelNet40/ply_data_train1.h5',
            'data/ModelNet40/ply_data_train2.h5',
            'data/ModelNet40/ply_data_train3.h5',
            'data/ModelNet40/ply_data_train4.h5'
        ]
    else:
        pathList = [
            'data/ModelNet40/ply_data_test0.h5',
            'data/ModelNet40/ply_data_test1.h5'
        ]
    all_data = []
    all_label = []
    for path in pathList:
        f = h5py.File(path,'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data,axis=0)
    all_label = np.concatenate(all_label,axis=0)
    return all_data,all_label



def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud,xyz1),xyz2).astype('float32')
    return translated_pointcloud

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    N, C = xyz.shape
    centroids = np.zeros(npoint, dtype=np.int32)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        # print("xyz[farthest]:",xyz[farthest])
        centroid = xyz[farthest].reshape(1,3)
        # print("centroid.shape:",centroid.shape)
        dist = np.sum((xyz - centroid) ** 2, -1)
        dists = np.zeros((2,N))
        dists[0] = dist
        dists[1] = distance
        # print("dists.shape:",dists.shape)
        distance = dists.min(axis=0)
        # print("distance.shape:",distance.shape)
        farthest = np.argmax(distance)
        print("farthest:",farthest)
    return centroids


class PointDataSet(Dataset):
    def __init__(self, num_points, mode='train', data_type='ScanObjectNN'):
        """
        加载数据集
        :param num_points: 1024
        :param mode: ['train', 'test']
        :param data_type: 数据集类型  ['ModelNet40', 'ScanObjectNN]'
        """
        self.num_points = num_points
        self.mode = mode
        if data_type == 'ScanObjectNN':
            self.data, self.label = load_scanobjectnn_data(mode)
        else:
            self.data, self.label = load_modelnet40_data(mode)

    def __getitem__(self, item):
        _, point_num, _ = self.data.shape
        all_points_index = np.arange(point_num)
        # replace=False, 采样的点都不一样
        # sample_points_index = np.random.choice(all_points_index, size=self.num_points, replace=False)  # 采样方式1
        sample_points_index = np.arange(self.num_points)  # 采样方式2
        # points = self.data[item]
        # sample_points_index = farthest_point_sample(points,self.num_points)  # 采样方式3
        # self.step = point_num // num_points   # 确保从点云中均匀取点，而不是只取前num_points个 或者random num_points个???
        pointcloud = self.data[item][sample_points_index]

        label = self.label[item]
        if self.mode == 'train':  # 是否需要做数据增强操作？？？由于随机采样，相当于已经对点云做了shuffle操作
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud,label

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    ds = PointDataSet(1024,mode='test',data_type='ModelNet40')
    # length = ds.__len__()
    # data,label = ds.__getitem__(0)
    # pts = vedo.Points(data)
    for data,label in ds:
        print("data.shape:",data.shape)
        print("label.shape:",label.shape)
    #     pts = vedo.Points(data)
    #     break
    # vedo.show(pts)
