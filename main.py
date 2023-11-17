import argparse
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import torch
import time
from model import SAHGCN,GCN,F2HGNN,utils

from loadData import split_data,data_read,data_reader
from createGraph import rdSLIC, create_graph
from createGraph.visualization import Draw_Classification_Map,visualize_hyperspectral_image
parser = argparse.ArgumentParser(description='SAHGCN')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()
def normalize_maxmin(Mx, axis=2):
    '''
    Normalize the matrix Mx by max-min normalization.
    axis=0: normalize each row
    axis=1: normalize each column
    axis=2: normalize the whole matrix
    '''
    Mx_min = Mx.min()
    if Mx_min < 0:
        Mx +=abs(Mx_min)
        Mx_min = Mx.min()

    if axis == 1:
        M_min = np.amin(Mx, axis=1)
        M_max = np.amax(Mx, axis=1)
        for i in range(Mx.shape[1]):
            Mx[:, i] = (Mx[:, i] - M_min) / (M_max - M_min)
    elif axis == 0:
        M_min = np.amin(Mx, axis=0)
        M_max = np.amax(Mx, axis=0)
        for i in range(Mx.shape[0]):
            Mx[i, :] = (Mx[i, :] - M_min) / (M_max - M_min)
    elif axis == 2:
        M_min = np.amin(Mx)
        M_max = np.amax(Mx)
        Mx = (Mx - M_min) / (M_max - M_min)
    else:
        print('Error')
        return None
    return Mx

data_set_name="paviaU"
# data_set_name="xuzhou"

# load data
print("loda_data now")
data,data_gt=data_read.readData(data_set_name)
#获取空間信息特征
get_spa=1
if get_spa==1:
    img=data
    gt=data_gt
    h, w = gt.shape[0], gt.shape[1]
    c = img.shape[2]
    idx = np.ones((img.shape[0], img.shape[1]))
    idx = np.where(idx == 1)
    idx_x = np.resize(idx[0], (img.shape[0], img.shape[1], 1))
    idx_y = np.resize(idx[1], (img.shape[0], img.shape[1], 1))
    img_idx = np.concatenate((idx_x, idx_y), axis=2)
    img_idx=normalize_maxmin(img_idx)

#設置采樣策略
samples_type="ratio" #ratio
# samples_type="number" #number
train_num=30
val_num=10
train_ratio=0.001
val_ratio=0.001
#選擇模型
mode_name="SAHGCN"
# mode_name="GCN"
# mode_name="F2HGNN"
if data_set_name=="paviaU":
    superpixel_scale = 100
else:
    superpixel_scale=100
max_epoch=400
Train_num=3
seed_list=[3]
learning_rate=0.001
weight_decay=0.001
lb_smooth=0.01

#要改
path_weight=r"D:\project\code\HGNN-base\SAHGCN\SAHGCN_main\weights"
path_result=r"D:\project\code\HGNN-base\SAHGCN\SAHGCN_main\results"

class_num = np.max(data_gt)
height, width, bands = data.shape
gt_reshape = np.reshape(data_gt, [-1])

torch.cuda.empty_cache()
OA_ALL = []
AA_ALL = []
KPP_ALL = []
AVG_ALL = []
Train_Time_ALL = []
Test_Time_ALL = []
#训练K次取平均值和方差
for i in range(Train_num):
    np.random.seed(seed_list[0])
    # split datasets
    train_index, val_index, test_index = split_data.split_data(gt_reshape,
                    class_num, train_ratio, val_ratio, train_num, val_num, samples_type)


    train_samples_gt, test_samples_gt, val_samples_gt = create_graph.get_label(gt_reshape,
                                                     train_index, val_index, test_index)


    train_label_mask, test_label_mask, val_label_mask = create_graph.get_label_mask(train_samples_gt,
                                                test_samples_gt, val_samples_gt, data_gt, class_num)

    # label transfer to one-hot encode
    train_gt = np.reshape(train_samples_gt,[height,width])
    test_gt = np.reshape(test_samples_gt,[height,width])
    val_gt = np.reshape(val_samples_gt,[height,width])
    print_data_info=1
    if print_data_info==1:
        data_reader.data_info(train_gt, val_gt, test_gt)

    train_gt_onehot = create_graph.label_to_one_hot(train_gt, class_num)
    test_gt_onehot = create_graph.label_to_one_hot(test_gt, class_num)
    val_gt_onehot = create_graph.label_to_one_hot(val_gt, class_num)

    print("LDA-SLIC Operation is Processing ")
    ls = rdSLIC.LDA_SLIC(data, train_gt, class_num-1)
    tic0=time.time()
    Q, S ,A, Seg= ls.simple_superpixel(scale=superpixel_scale)
    toc0 = time.time()
    LDA_SLIC_Time=toc0-tic0
    Q=torch.from_numpy(Q).to(args.device)
    A=torch.from_numpy(A).to(args.device)

    train_samples_gt=torch.from_numpy(train_samples_gt.astype(np.float32)).to(args.device)
    test_samples_gt=torch.from_numpy(test_samples_gt.astype(np.float32)).to(args.device)
    val_samples_gt=torch.from_numpy(val_samples_gt.astype(np.float32)).to(args.device)

    train_gt_onehot = torch.from_numpy(train_gt_onehot.astype(np.float32)).to(args.device)
    test_gt_onehot = torch.from_numpy(test_gt_onehot.astype(np.float32)).to(args.device)
    val_gt_onehot = torch.from_numpy(val_gt_onehot.astype(np.float32)).to(args.device)

    train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(args.device)
    test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(args.device)
    val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(args.device)

    #聯合光譜-空間信息
    net_input = np.array(data, np.float32)
    net_input=normalize_maxmin(net_input)
    net_input=net_input = np.concatenate((net_input, img_idx), axis=2)
    net_input = torch.from_numpy(net_input.astype(np.float32)).to(args.device)

    # model
    if mode_name=="SAHGCN":
        net=SAHGCN.SAHGCN(height, width, bands+2, class_num, Q, A,alph=0.1).to(args.device)
    elif mode_name=="GCN":
        net=GCN.GCN(bands,128,class_num,Q,A).to(args.device)
    elif mode_name=="F2HGNN":
        net=F2HGNN.HGNN_weight(net_input.shape[-1],256,class_num,Q,A,dropout=0.2).to(args.device)

    # train
    print("\n\n==================== train ====================\n")
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate, weight_decay=weight_decay) #, weight_decay=0.0001
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    zeros = torch.zeros([height * width]
                        ).to(args.device).float()
    best_loss=99999
    net.train()
    tic1 = time.time()
    for i in range(max_epoch+1):
        optimizer.zero_grad()  # zero the gradient buffers
        output= net(net_input)
        loss = utils.compute_loss(output, train_gt_onehot, train_label_mask)
        loss.backward(retain_graph=False)
        optimizer.step()  # Does the update

        # if i%10==0:
        with torch.no_grad():
            net.eval()
            output= net(net_input)
            trainloss = utils.compute_loss(output, train_gt_onehot, train_label_mask)
            trainOA = utils.evaluate_performance(output, train_samples_gt, train_gt_onehot, zeros)
            valloss = utils.compute_loss(output, val_gt_onehot, val_label_mask)
            valOA = utils.evaluate_performance(output, val_samples_gt, val_gt_onehot, zeros)
            # print("{}\ttrain loss={:.4f}\t train OA={:.4f} val loss={:.4f}\t val OA={:.4f}".format(str(i + 1), trainloss, trainOA, valloss, valOA))

            if valloss < best_loss :
                best_loss = valloss
                torch.save(net.state_dict(), path_weight + r"model.pt")
                print('save model...')
        scheduler.step(valloss)
        torch.cuda.empty_cache()
        net.train()

        if i%10==0:
            print("{}\ttrain loss={:.4f}\t train OA={:.4f} val loss={:.4f}\t val OA={:.4f}".format(str(i + 1), trainloss, trainOA, valloss, valOA))
    toc1 = time.time()

    print("\n\n====================training done. starting evaluation...========================\n")

    # test
    torch.cuda.empty_cache()
    with torch.no_grad():
        net.load_state_dict(torch.load(path_weight + r"model.pt"))
        net.eval()
        tic2 = time.time()
        output = net(net_input)
        toc2 = time.time()
        testloss = utils.compute_loss(output, test_gt_onehot, test_label_mask)
        print(zeros.shape)
        print(class_num)
        print(height)
        print(width)
        testOA,testAA,testKppa,test_AC_list = utils.evaluate_performance2(output, test_samples_gt, test_gt_onehot, zeros,np.array(class_num),np.array(height),np.array(width))
        print("{}\ttest loss={:.4f}\t test OA={:.4f}\t test AA={:.4f}\t test kppa={:.4f}".format(str(i + 1), testloss, testOA,testAA,testKppa))
        OA_ALL.append(testOA)
        AA_ALL.append(testAA)
        KPP_ALL.append(testKppa)
        AVG_ALL.append(test_AC_list)
    torch.cuda.empty_cache()
    del net

    LDA_SLIC_Time=toc0-tic0
    # print("LDA-SLIC costs time: {}".format(LDA_SLIC_Time))
    training_time = toc1 - tic1 + LDA_SLIC_Time
    testing_time = toc2 - tic2 + LDA_SLIC_Time
    training_time, testing_time

    # classification report
    test_label_mask_cpu = test_label_mask.cpu().numpy()[:,0].astype('bool')
    test_samples_gt_cpu = test_samples_gt.cpu().numpy().astype('int64')
    predict = torch.argmax(output, 1).cpu().numpy()

    classification_map = torch.argmax(output, 1).reshape([height, width]).cpu() + 1
    mask=data_gt>0
    masked_predicted_labels = np.copy(classification_map)
    masked_predicted_labels[~mask]=0
    # visualize_hyperspectral_image(masked_predicted_labels)
    Draw_Classification_Map(classification_map, "results\\Image\\" +mode_name+"-"+ data_set_name + str(testOA)+"CVPR_100"+"+SAH")
    Draw_Classification_Map(masked_predicted_labels, "results\\Mask\\" + mode_name+"-"+data_set_name + str(testOA)+"CVPR_100"+"+SAH")

    classification = classification_report(test_samples_gt_cpu[test_label_mask_cpu],
                                        predict[test_label_mask_cpu]+1, digits=4)
    kappa = cohen_kappa_score(test_samples_gt_cpu[test_label_mask_cpu], predict[test_label_mask_cpu]+1)
    show_results=1
    if show_results==1:
        print(classification, kappa)

    # store results
    save_results=1
    if save_results==1:
        print("save results")
        run_date = time.strftime('%Y%m%d-%H%M-',time.localtime(time.time()))
        f = open(path_result + run_date +mode_name+"-"+data_set_name +"CVPR"+ '.txt', 'a+')
        str_results = '\n ======================' \
                    + '\nrun data = ' + run_date \
                    + "\nlearning rate = " + str(learning_rate) \
                    + "\nepochs = " + str(max_epoch) \
                    + "\nsamples_type = " + str(samples_type) \
                    + "\ntrain ratio = " + str(train_ratio) \
                    + "\nval ratio = " + str(val_ratio) \
                    + "\ntrain num = " + str(train_num) \
                    + "\nval num = " + str(val_num) \
                    + '\ntrain time = ' + str(training_time) \
                    + '\ntest time = ' + str(testing_time) \
                    + '\n' + classification \
                    + "kappa = " + str(kappa) \
                    + '\n'
        f.write(str_results)
        f.close()

OA_ALL = np.array(OA_ALL)
AA_ALL = np.array(AA_ALL)
KPP_ALL = np.array(KPP_ALL)
AVG_ALL = np.array(AVG_ALL)
Train_Time_ALL=np.array(Train_Time_ALL)
Test_Time_ALL=np.array(Test_Time_ALL)
print("")
print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
