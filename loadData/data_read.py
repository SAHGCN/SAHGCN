import h5py
import scipy.io as scio
import numpy as np

def readData(data_set_name):
    if data_set_name == "Indian":
        img = scio.loadmat(r'D:\dataset\Indian/Indian_pines_corrected.mat')['indian_pines_corrected']
        gt = scio.loadmat(r'D:\dataset\Indian/Indian_pines_gt.mat')['indian_pines_gt']

    if data_set_name == "paviaU":
        img = scio.loadmat(r'D:\dataset\高光谱数据集\高光谱数据集\Pavia\PaviaU.mat')["paviaU"]
        gt = scio.loadmat(r'D:\dataset\高光谱数据集\高光谱数据集\Pavia\PaviaU_gt.mat')["Data_gt"]
    if data_set_name == "salinas":
        img = scio.loadmat(r'D:\dataset\高光谱数据集\高光谱数据集\Salinas\salinas.mat')["HSI_original"]
        gt = scio.loadmat(r'D:\dataset\高光谱数据集\高光谱数据集\Salinas\salinas_gt.mat')["Data_gt"]

    if data_set_name == "KSC":
        img = scio.loadmat(r'D:\dataset\高光谱数据集\高光谱数据集\KSC\KSC.mat')['KSC']
        gt = scio.loadmat(r'D:\dataset\高光谱数据集\高光谱数据集\KSC\KSC_gt.mat')['KSC_gt']
    if data_set_name == "HoustonU":
        img = h5py.File(r'D:\dataset\高光谱数据集\高光谱数据集\HoustonU\HoustonU.mat', "r")
        img = np.array(img.get("houstonU"))
        img = np.transpose(img, (1, 2, 0))
        gt = h5py.File(r'D:\dataset\高光谱数据集\高光谱数据集\HoustonU\HoustonU_gt.mat', "r")
        gt = np.array(gt.get("houstonU_gt"))

    if data_set_name == "Houston2013":
        img = scio.loadmat(r'D:\dataset\高光谱数据集\高光谱数据集\HoustonU\Houston.mat')["Houston"]
        gt = scio.loadmat(r'D:\dataset\高光谱数据集\高光谱数据集\HoustonU\Houston_gt.mat')["Houston_gt"]

    if data_set_name == "xiongan":
        img = h5py.File(r"D:\dataset\高光谱数据集\高光谱数据集\XiongAn\xiongan.mat", "r")
        key1 = img.keys()
        img = np.array(img.get("XiongAn"))
        img = np.transpose(img, (1, 2, 0))
        gt = h5py.File(r'D:\dataset\高光谱数据集\高光谱数据集\XiongAn\xiongan_gt.mat', "r")

        key2 = gt.keys()
        gt = np.array(gt.get("xiongan_gt"))

    if data_set_name == "xuzhou":
        img = scio.loadmat(r"D:\dataset\高光谱数据集\高光谱数据集\Xuzhou\xuzhou.mat")["xuzhou"]
        gt = scio.loadmat(r"D:\dataset\高光谱数据集\高光谱数据集\\Xuzhou\xuzhou_gt.mat")["xuzhou_gt"]
    return img,gt

