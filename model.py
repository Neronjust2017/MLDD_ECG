import os

import numpy as np
import torch
import torch.utils.model_zoo as model_zoo
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch.optim import lr_scheduler
import pickle as dill

import cnn
from data_reader import BatchEcgGenerator
from utils import sgd, crossentropyloss, fix_seed, write_log, compute_accuracy


class ModelBaseline:
    def __init__(self, flags):

        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # fix the random seed or not
        fix_seed()

        self.setup_dataset(flags)

        self.network = cnn.CNNNet(num_classes=flags.num_classes)

        self.network = self.network.cuda()

        print(self.network)
        print('flags:', flags)

        if not os.path.exists(flags.logs):
            os.mkdir(flags.logs)

        flags_log = os.path.join(flags.logs, 'flags_log.txt')
        write_log(flags, flags_log)

        self.load_state_dict(flags.state_dict)

        self.configure(flags)

    def setup_dataset(self, flags):

        root_folder = flags.data_root
        unseen_index = flags.unseen_index
        val_split = flags.val_split

        with open(os.path.join(root_folder, 'af_normal_data_processed.pkl'), 'rb') as file:
            data = dill.load(file)

        datasets = ['CSPC_data', 'PTB_XL_data', 'G12EC_data', 'Challenge2017_data']

        test_data = []
        train_datas = []
        val_datas = []

        for source in datasets:
            af_data, normal_data = data[source]
            all_data = np.concatenate((af_data, normal_data), axis=0)
            all_label = np.zeros((len(all_data), ))
            all_label[len(af_data):] = 1

            # use all data of this source as test data
            permuted_idx = np.random.permutation(len(all_data))
            x = all_data[permuted_idx]
            y = all_label[permuted_idx]

            split_idx = int(val_split * len(all_data))
            x_val = all_data[permuted_idx[split_idx:]]
            y_val = all_label[permuted_idx[split_idx:]]
            x_train = all_data[permuted_idx[:split_idx]]
            y_train = all_label[permuted_idx[:split_idx]]

            # swap axes
            x = x.swapaxes(1,2)
            x_train = x_train.swapaxes(1,2)
            x_val = x_val.swapaxes(1,2)

            test_data.append([x, y])
            train_datas.append([x_train, y_train])
            val_datas.append([x_val, y_val])

        self.train_datas = train_datas
        self.val_datas = val_datas

        a = [0, 1, 2, 3]
        a.remove(unseen_index)

        self.unseen_data = test_data[unseen_index]
        del self.train_datas[unseen_index]
        del self.val_datas[unseen_index]

        if not os.path.exists(flags.logs):
            # os.mkdir(flags.logs)
            os.makedirs(flags.logs)

        self.batEcgGenTrains = []
        for train_data in self.train_datas:
            batEcgGenTrain = BatchEcgGenerator(flags=flags, dataset=train_data, stage='train',
                                                   b_unfold_label=False)
            self.batEcgGenTrains.append(batEcgGenTrain)

        self.batEcgGenVals = []
        for val_data in self.val_datas:
            batEcgGenVal = BatchEcgGenerator(flags=flags, dataset=val_data, stage='val',
                                                 b_unfold_label=True)
            self.batEcgGenVals.append(batEcgGenVal)

    def load_state_dict(self, state_dict=''):

        if state_dict:
            try:
                tmp = torch.load(state_dict)
                pretrained_dict = tmp['state']
            except:
                pretrained_dict = model_zoo.load_url(state_dict)

            model_dict = self.network.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and v.size() == model_dict[k].size()}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.network.load_state_dict(model_dict)

    def heldout_test(self, flags):

        # load the best model in the validation data
        model_path = os.path.join(flags.model_path, 'best_model.tar')
        self.load_state_dict(state_dict=model_path)

        # test
        batEcgGenTest = BatchEcgGenerator(flags=flags, dataset=self.unseen_data, stage='test',
                                              b_unfold_label=False)
        test_ecgs = batEcgGenTest.ecgs

        threshold = 100
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
        self.network.eval()
        for test_ecg_split in test_ecg_splits:
            ecgs_test = Variable(torch.from_numpy(np.array(test_ecg_split, dtype=np.float32))).cuda()
            outputs, end_points = self.network(ecgs_test)

            pred = end_points['Predictions']
            pred = pred.cpu().data.numpy()
            predictions.append(pred)

        # concatenate the test predictions first
        predictions = np.concatenate(predictions)

        # accuracy
        accuracy = accuracy_score(y_true=batEcgGenTest.labels,
                                  y_pred=np.argmax(predictions, -1))

        flags_log = os.path.join(flags.logs, 'heldout_test_log.txt')
        write_log(accuracy, flags_log)

    def configure(self, flags):

        for name, para in self.network.named_parameters():
            print(name, para.size())

        # self.optimizer = sgd(parameters=self.network.parameters(),
        #                      lr=flags.lr,
        #                      weight_decay=flags.weight_decay,
        #                      momentum=flags.momentum)
        self.optimizer = sgd(parameters=self.network.parameters(),
                             lr=flags.lr,
                             weight_decay=flags.weight_decay,
                             momentum=flags.momentum)

        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=flags.step_size, gamma=0.1)
        self.loss_fn = crossentropyloss()

    def train(self, flags):
        self.network.train()

        self.best_accuracy_val = -1

        for ite in range(flags.inner_loops):

            self.scheduler.step(epoch=ite)

            total_loss = 0.0
            for index in range(len(self.batEcgGenTrains)):
                ecgs_train, labels_train = self.batEcgGenTrains[index].get_ecgs_labels_batch()

                inputs, labels = torch.from_numpy(
                    np.array(ecgs_train, dtype=np.float32)), torch.from_numpy(
                    np.array(labels_train, dtype=np.float32))

                # wrap the inputs and labels in Variable
                inputs, labels = Variable(inputs, requires_grad=False).cuda(), \
                                 Variable(labels, requires_grad=False).long().cuda()

                outputs, _ = self.network(x=inputs)

                # loss
                loss = self.loss_fn(outputs, labels)
                total_loss += loss

            # init the grad to zeros first
            self.optimizer.zero_grad()

            # backward your network
            total_loss.backward()

            # optimize the parameters
            self.optimizer.step()

            print(
                'ite:', ite, 'loss:', total_loss.cpu().item(), 'lr:',
                self.scheduler.get_lr()[0])

            flags_log = os.path.join(flags.logs, 'loss_log.txt')
            write_log(
                str(total_loss.cpu().item()),
                flags_log)

            del total_loss, outputs

            if ite % flags.test_every == 0 and ite is not 0 or flags.debug:
                self.test_workflow(self.batEcgGenVals, flags, ite)

    def test_workflow(self, batEcgGenVals, flags, ite):

        accuracies = []
        for count, batEcgGenVal in enumerate(batEcgGenVals):

            ## test on validation
            accuracy_val = self.test(batEcgGenTest=batEcgGenVal, flags=flags, ite=ite,
                                     log_dir=flags.logs, log_prefix='val_index_{}'.format(count))

            accuracies.append(accuracy_val)

        mean_acc = np.mean(accuracies)

        if mean_acc > self.best_accuracy_val:
            self.best_accuracy_val = mean_acc

            f = open(os.path.join(flags.logs, 'Best_val.txt'), mode='a')
            f.write('ite:{}, best val accuracy:{}\n'.format(ite, self.best_accuracy_val))
            f.close()

            if not os.path.exists(flags.model_path):
                os.mkdir(flags.model_path)

            outfile = os.path.join(flags.model_path, 'best_model.tar')
            torch.save({'ite': ite, 'state': self.network.state_dict()}, outfile)

    def test(self, flags, ite, log_prefix, log_dir='logs/', batEcgGenTest=None):

        # switch on the network test mode
        self.network.eval()

        if batEcgGenTest is None:
            batEcgGenTest = BatchEcgGenerator(flags=flags, dataset=None, stage='test', b_unfold_label=True)

        ecgs_test = batEcgGenTest.ecgs
        labels_test = batEcgGenTest.labels

        threshold = 50
        if len(ecgs_test) > threshold:

            n_slices_test = int(len(ecgs_test) / threshold) + 1
            indices_test = []
            for per_slice in range(n_slices_test - 1):
                indices_test.append(int(len(ecgs_test) * (per_slice + 1) / n_slices_test))
            test_ecg_splits = np.split(ecgs_test, indices_or_sections=indices_test)

            # Verify the splits are correct
            test_ecg_splits_2_whole = np.concatenate(test_ecg_splits)
            assert np.all(ecgs_test == test_ecg_splits_2_whole)

            # split the test data into splits and test them one by one
            test_ecg_preds = []
            for test_ecg_split in test_ecg_splits:
                ecgs_test = Variable(torch.from_numpy(np.array(test_ecg_split, dtype=np.float32))).cuda()
                outputs, end_points = self.network(ecgs_test)

                predictions = end_points['Predictions']
                predictions = predictions.cpu().data.numpy()
                test_ecg_preds.append(predictions)

            # concatenate the test predictions first
            predictions = np.concatenate(test_ecg_preds)
        else:
            ecgs_test = Variable(torch.from_numpy(np.array(ecgs_test, dtype=np.float32))).cuda()
            outputs, end_points = self.network(ecgs_test)

            predictions = end_points['Predictions']
            predictions = predictions.cpu().data.numpy()

        accuracy = compute_accuracy(predictions=predictions, labels=labels_test)
        print('----------accuracy test----------:', accuracy)

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        log_path = os.path.join(log_dir, '{}.txt'.format(log_prefix))
        write_log(str('ite:{}, accuracy:{}'.format(ite, accuracy)), log_path=log_path)

        # switch on the network train mode after test
        self.network.train()

        return accuracy


class ModelMLDG(ModelBaseline):
    def __init__(self, flags):

        ModelBaseline.__init__(self, flags)

    def train(self, flags):
        self.network.train()

        self.best_accuracy_val = -1

        for ite in range(flags.inner_loops):

            self.scheduler.step(epoch=ite)

            # select the validation domain for meta val
            index_val = np.random.choice(a=np.arange(0, len(self.batEcgGenTrains)), size=1)[0]
            batEcgMetaVal = self.batEcgGenTrains[index_val]

            meta_train_loss = 0.0
            # get the inputs and labels from the data reader
            for index in range(len(self.batEcgGenTrains)):

                if index == index_val:
                    continue

                ecgs_train, labels_train = self.batEcgGenTrains[index].get_ecgs_labels_batch()

                inputs_train, labels_train = torch.from_numpy(
                    np.array(ecgs_train, dtype=np.float32)), torch.from_numpy(
                    np.array(labels_train, dtype=np.float32))

                # wrap the inputs and labels in Variable
                inputs_train, labels_train = Variable(inputs_train, requires_grad=False).cuda(), \
                                             Variable(labels_train, requires_grad=False).long().cuda()

                # forward with the adapted parameters
                outputs_train, _ = self.network(x=inputs_train)

                # loss
                loss = self.loss_fn(outputs_train, labels_train)
                meta_train_loss += loss

            ecg_val, labels_val = batEcgMetaVal.get_ecgs_labels_batch()
            inputs_val, labels_val = torch.from_numpy(
                np.array(ecg_val, dtype=np.float32)), torch.from_numpy(
                np.array(labels_val, dtype=np.float32))

            # wrap the inputs and labels in Variable
            inputs_val, labels_val = Variable(inputs_val, requires_grad=False).cuda(), \
                                     Variable(labels_val, requires_grad=False).long().cuda()

            # forward with the adapted parameters
            outputs_val, _ = self.network(x=inputs_val,
                                          meta_loss=meta_train_loss,
                                          meta_step_size=flags.meta_step_size,
                                          stop_gradient=flags.stop_gradient)

            meta_val_loss = self.loss_fn(outputs_val, labels_val)

            total_loss = meta_train_loss + meta_val_loss * flags.meta_val_beta

            # init the grad to zeros first
            self.optimizer.zero_grad()

            # backward your network
            total_loss.backward()

            # optimize the parameters
            self.optimizer.step()

            print(
                'ite:', ite,
                'meta_train_loss:', meta_train_loss.cpu().item(),
                'meta_val_loss:', meta_val_loss.cpu().item(),
                'lr:',
                self.scheduler.get_lr()[0])

            flags_log = os.path.join(flags.logs, 'loss_log.txt')
            write_log(
                str(meta_train_loss.cpu().item()) + '\t' + str(meta_val_loss.cpu().item()),
                flags_log)

            del total_loss, outputs_val, outputs_train

            if ite % flags.test_every == 0 and ite is not 0 or flags.debug:
                self.test_workflow(self.batEcgGenVals, flags, ite)
