import torch
import torch.backends.cudnn as cudnn

from nets.arcface import Arcface
from utils.dataloader import LFWDataset
from utils.utils_metrics import test





if __name__ == "__main__":
    #--------------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #--------------------------------------#
    cuda            = True
    #--------------------------------------#
    #   主干特征提取网络的选择
    #   mobilefacenet
    #   mobilenetv1
    #   iresnet18
    #   iresnet34
    #   iresnet50
    #   iresnet100
    #   iresnet200
    #--------------------------------------#
    backbone        = "mobilefacenet"
    #--------------------------------------#
    #   输入图像大小
    #--------------------------------------#
    input_shape     = [112, 112, 3]
    #--------------------------------------#
    #   训练好的权值文件
    #--------------------------------------#
    model_path      = "/home/qiujing/cqwork/arcface-pytorch/model_data/mobilenet_v1_backbone_weights.pth"     # "model_data/arcface_mobilefacenet.pth"    
    # "/home/qiujing/cqwork/arcface-pytorch/model_data/mobilenet_v1_backbone_weights.pth"
    # /home/qiujing/cqwork/arcface-pytorch/model_data/arcface_mobilefacenet.pth
    #--------------------------------------#
    #   LFW评估数据集的文件路径
    #   以及对应的txt文件
    #--------------------------------------#
    lfw_dir_path    = "/mnt/ssd/qiujing/arcface/lfw/lfw112/112×112"      # "lfw"     # /home/qiujing/cqwork/arcface-pytorch/lfw      
    # /mnt/ssd/qiujing/arcface/lfw/lfw112/112×112
    # /mnt/ssd/qiujing/arcface/cfp/cfp/cfp-dataset/Data/Images ，需添加LFWDataset部分
    # /mnt/ssd/qiujing/arcface/megafacetest/megaface_images，有专门的评估工具devkit.tar.gz， 已截止


    lfw_pairs_path  = "/mnt/ssd/qiujing/arcface/lfw/lfw112/pairs.txt"     # "model_data/lfw_pair.txt"   # /home/qiujing/cqwork/arcface-pytorch/model_data/lfw_pair.txt      
    # /mnt/ssd/qiujing/arcface/lfw/lfw112/pairs.txt
    # /mnt/ssd/qiujing/arcface/cfp/cfp/cfp-dataset/Protocol/Pair_list_P.txt  ，需添加LFWDataset部分
    # /mnt/ssd/qiujing/arcface/megafacetest/megaface_noises.txt，有专门的评估工具devkit.tar.gz， 已截止

    #--------------------------------------#
    #   评估的批次大小和记录间隔
    #--------------------------------------#
    batch_size      = 256
    log_interval    = 1
    #--------------------------------------#
    #   ROC图的保存路径
    #--------------------------------------#
    png_save_path   = "/home/qiujing/cqwork/arcface-pytorch/model_data/roc_test.png"   # "model_data/roc_test.png"     # /home/qiujing/cqwork/arcface-pytorch/model_data/roc_test.png 

    test_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=batch_size, shuffle=False)

    model = Arcface(backbone=backbone, mode="predict")

    print('Loading weights into state dict...')   # 加载权重到状态字典
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model  = model.eval()

    if cuda:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()

    test(test_loader, model, png_save_path, log_interval, batch_size, cuda)
