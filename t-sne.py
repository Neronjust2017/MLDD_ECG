import os
import numpy as np
import torch
import pickle as dill
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

from cnn import CNNNet

def load_state_dict(network, state_dict=''):

    if state_dict:
        try:
            tmp = torch.load(state_dict)
            pretrained_dict = tmp['state']
        except:
            pretrained_dict = model_zoo.load_url(state_dict)

        model_dict = network.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.size() == model_dict[k].size()}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        network.load_state_dict(model_dict)

def load_data(root_folder):

    with open(os.path.join(root_folder, 'af_normal_data_processed.pkl'), 'rb') as file:
        data = dill.load(file)

    datasets = ['CSPC_data', 'PTB_XL_data', 'G12EC_data', 'Challenge2017_data']

    datas = []

    for source in datasets:
        af_data, normal_data = data[source]
        all_data = np.concatenate((af_data, normal_data), axis=0)
        all_label = np.zeros((len(all_data),))
        all_label[len(af_data):] = 1

        all_data = all_data.swapaxes(1,2)
        datas.append([all_data, all_label])

    return datas




if __name__ == '__main__':

    unseen_index = 0
    run_path = 'run_1'
    model_path = run_path+'/models_'+str(unseen_index) + '/best_model.tar'
    data_path = '/home/weiyuhua/Challenge2020/Data/DG'

    # model
    network = CNNNet()
    print(network)
    load_state_dict(network, model_path)

    # for name, m in network.named_modules():
    #     print(name)
    # conv1 conv2 conv3 pooling fc1 fc2 fc3 bn1 bn2 bn3 dropout act

    # data
    datas = load_data(data_path)
    unseen_data = datas[unseen_index]
    del datas[unseen_index]

    # gpu
    cuda = False
    device = 'cpu'
    if cuda and torch.cuda.is_available():
        device = 'cuda'

    # 中间层输出
    # class LayerActivations:
    #     features = None
    #
    #     def __init__(self, model, layer_num):
    #         self.hook = model[layer_num].register_forward_hook(self.hook_fn)
    #
    #     def hook_fn(self, module, input, output):
    #         self.features = output.cpu()
    #
    #     def remove(self):
    #         self.hook.remove()

    # Inferance

    for data in datas:

        test_ecgs = data[0]

        threshold = 1000
        n_slices_test = int(len(test_ecgs) / threshold) + 1
        indices_test = []
        for per_slice in range(n_slices_test - 1):
            indices_test.append(int(len(test_ecgs) * (per_slice + 1) / n_slices_test))
        test_ecg_splits = np.split(test_ecgs, indices_or_sections=indices_test)

        # Verify the splits are correct
        test_ecg_splits_2_whole = np.concatenate(test_ecg_splits)
        assert np.all(test_ecgs == test_ecg_splits_2_whole)

        # split the test data into splits and test them one by one
        predictions = []
        network.eval()
        for test_ecg_split in test_ecg_splits:
            ecgs_test = Variable(torch.from_numpy(np.array(test_ecg_split, dtype=np.float32))).to(device)
            outputs, end_points = network.fc1(ecgs_test)

            pred = end_points['Predictions']
            pred = pred.cpu().data.numpy()
            predictions.append(pred)

        print(2)






