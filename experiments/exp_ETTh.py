import os
import time

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')
from data_process.etth_data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, save_model, load_model
from metrics.ETTh_metrics import metric
from models.SCINet import SCINet
from models.SCINet_decompose import SCINet_decompose

def UncertaintyLoss(logits, ground_truths):
    print(f"logits shape: {logits.shape}")
    print(f"ground_truths shape: {ground_truths.shape}")
    return 0

"""
class UncertaintyLoss():
    def get_loss(logits, ground_truths):
        multi_loss_class = None
        loss_list = []
        if FLAGS.use_label_type:
            if FLAGS.need_resize:
                label_type = tf.image.resize_images(ground_truths[0], [FLAGS.output_height, FLAGS.output_width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
            else:
                label_type = ground_truths[0]
            loss_list.append(loss(logits[0], label_type, type='cross_entropy'))
        if FLAGS.use_label_inst:
            xy_gt = tf.slice(ground_truths[1], [0, 0, 0, 0], [-1, FLAGS.output_height, FLAGS.output_width, 2])    # to get x GT and y GT
            mask = tf.slice(ground_truths[1], [0, 0, 0, 2], [-1, FLAGS.output_height, FLAGS.output_width, 1])  # to get mask from GT
            mask = tf.concat([mask, mask], 3)  # to get mask for x and for y
            if FLAGS.need_resize:
                xy_gt = tf.image.resize_images(xy_gt, [FLAGS.output_height, FLAGS.output_width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
                mask = tf.image.resize_images(mask, [FLAGS.output_height, FLAGS.output_width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
            loss_list.append(l1_masked_loss(tf.multiply(logits[1], mask), xy_gt, mask))
        if FLAGS.use_label_disp:
            if FLAGS.need_resize:
                gt_sized = tf.image.resize_images(ground_truths[2], [FLAGS.output_height, FLAGS.output_width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
                gt_sized = gt_sized[:, :, :, 0]
                mask = gt_sized[:, :, :, 1]
            else:
                gt_sized = tf.expand_dims(ground_truths[2][:, :, :, 0], axis=-1)
                mask = tf.expand_dims(ground_truths[2][:, :, :, 1], axis=-1)
            loss_list.append(l1_masked_loss(tf.multiply(logits[2], mask), tf.multiply(gt_sized, mask), mask))
        if FLAGS.use_multi_loss:
            loss_op, multi_loss_class = calc_multi_loss(loss_list)
        else:
            loss_op = loss_list[0]
            for i in range(1, len(loss_list)):
                loss_op = tf.add(loss_op, loss_list[i])
        return loss_op, loss_list, multi_loss_class
    def calc_multi_loss(loss_list):
        multi_loss_layer = MultiLossLayer(loss_list)
        return multi_loss_layer.get_loss(), multi_loss_layer
"""
class Exp_ETTh(Exp_Basic):
    def __init__(self, args):
        super(Exp_ETTh, self).__init__(args)
    def _build_model(self):

        if self.args.features == 'S':
            in_dim = 1
        elif self.args.features == 'M':
            in_dim = 21
            if self.args.data == 'ALL' or self.args.data == 'ALL_CUT':
                in_dim = 17
            elif self.args.data == 'No_MonthSinCos' or self.args.data == 'No_MonthSinCos_CUT':
                in_dim = 15
            elif self.args.data == 'No_EFA_MonthSinCos' or self.args.data == 'No_EFA_MonthSinCos_CUT':
                in_dim = 13
            elif self.args.data == 'No_EFA_Day_MonthSinCos' or self.args.data == 'No_EFA_Day_MonthSinCos_CUT':
                in_dim = 12
            elif self.args.data == 'July_Important_Variable' or self.args.data == 'July_Important_Variable_CUT':
                in_dim = 11
            elif self.args.data == 'July_High_Correlation_Table' or self.args.data == 'July_High_Correlation_Table_CUT':
                in_dim = 8
            elif self.args.data == 'ALL_July' or self.args.data == 'ALL_CUT_July':
                in_dim = 17
            elif self.args.data == 'No_MonthSinCos_July' or self.args.data == 'No_MonthSinCos_CUT_July':
                in_dim = 15
            elif self.args.data == 'No_EFA_MonthSinCos_July' or self.args.data == 'No_EFA_MonthSinCos_CUT_July':
                in_dim = 13
            elif self.args.data == 'No_EFA_Day_MonthSinCos_July' or self.args.data == 'No_EFA_Day_MonthSinCos_CUT_July':
                in_dim = 12
            elif self.args.data == 'July_Important_Variable_July' or self.args.data == 'July_Important_Variable_CUT_July':
                in_dim = 11
            elif self.args.data == 'July_High_Correlation_Table_July' or self.args.data == 'July_High_Correlation_Table_CUT_July':
                in_dim = 8
            elif self.args.data == 'ALL_TEMP' or self.args.data == 'ALL_CUT_TEMP':
                in_dim = 18
            elif self.args.data == 'July_High_Correlation_Table_TEMP' or self.args.data == 'July_High_Correlation_Table_CUT_TEMP':
                in_dim = 9
            elif self.args.data == 'ALL_TEMP_WIND' or self.args.data == 'ALL_CUT_TEMP_WIND':
                in_dim = 19
            elif self.args.data == 'July_High_Correlation_Table_TEMP_WIND' or self.args.data == 'July_High_Correlation_Table_CUT_TEMP_WIND':
                in_dim = 10
            elif self.args.data == 'August_High_Correlation_TEMP_WIND':
                in_dim = 10
            elif self.args.data == 'August_High_Correlation_TEMP':
                in_dim = 9
            # in_dim = 22
            # if self.args.data == 'ALL' or self.args.data == 'ALL_CUT':
            #     in_dim = 18
            # elif self.args.data == 'No_MonthSinCos' or self.args.data == 'No_MonthSinCos_CUT':
            #     in_dim = 16
            # elif self.args.data == 'No_EFA_MonthSinCos' or self.args.data == 'No_EFA_MonthSinCos_CUT':
            #     in_dim = 14
            # elif self.args.data == 'No_EFA_Day_MonthSinCos' or self.args.data == 'No_EFA_Day_MonthSinCos_CUT':
            #     in_dim = 13
            # elif self.args.data == 'July_Important_Variable' or self.args.data == 'July_Important_Variable_CUT':
            #     in_dim = 12
            # elif self.args.data == 'July_High_Correlation_Table' or self.args.data == 'July_High_Correlation_Table_CUT':
            #     in_dim = 9
            # elif self.args.data == 'ALL_July' or self.args.data == 'ALL_CUT_July':
            #     in_dim = 18
            # elif self.args.data == 'No_MonthSinCos_July' or self.args.data == 'No_MonthSinCos_CUT_July':
            #     in_dim = 16
            # elif self.args.data == 'No_EFA_MonthSinCos_July' or self.args.data == 'No_EFA_MonthSinCos_CUT_July':
            #     in_dim = 14
            # elif self.args.data == 'No_EFA_Day_MonthSinCos_July' or self.args.data == 'No_EFA_Day_MonthSinCos_CUT_July':
            #     in_dim = 13
            # elif self.args.data == 'July_Important_Variable_July' or self.args.data == 'July_Important_Variable_CUT_July':
            #     in_dim = 12
            # elif self.args.data == 'July_High_Correlation_Table_July' or self.args.data == 'July_High_Correlation_Table_CUT_July':
            #     in_dim = 9
        else:
            print('Error!')

        if self.args.decompose:
            model = SCINet_decompose(
                output_len=self.args.pred_len,
                input_len=self.args.seq_len,
                input_dim= in_dim,
                hid_size = self.args.hidden_size,
                num_stacks=self.args.stacks,
                num_levels=self.args.levels,
                num_decoder_layer=self.args.num_decoder_layer,
                concat_len = self.args.concat_len,
                groups = self.args.groups,
                kernel = self.args.kernel,
                dropout = self.args.dropout,
                single_step_output_One = self.args.single_step_output_One,
                positionalE = self.args.positionalEcoding,
                modified = True,
                RIN=self.args.RIN)
        else:
            model = SCINet(
                output_len=self.args.pred_len,
                input_len=self.args.seq_len,
                input_dim= in_dim,
                hid_size = self.args.hidden_size,
                num_stacks=self.args.stacks,
                num_levels=self.args.levels,
                num_decoder_layer=self.args.num_decoder_layer,
                concat_len = self.args.concat_len,
                groups = self.args.groups,
                kernel = self.args.kernel,
                dropout = self.args.dropout,
                single_step_output_One = self.args.single_step_output_One,
                positionalE = self.args.positionalEcoding,
                modified = True,
                RIN=self.args.RIN)
        print(model)
        return model.double()

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
            'delta': Dataset_Custom,
            'ALL': Dataset_Custom,
            'No_MonthSinCos': Dataset_Custom, 
            'No_EFA_MonthSinCos': Dataset_Custom,
            'No_EFA_Day_MonthSinCos': Dataset_Custom,
            'July_Important_Variable': Dataset_Custom,
            'July_High_Correlation_Table': Dataset_Custom,
            'ALL_CUT': Dataset_Custom,
            'No_MonthSinCos_CUT': Dataset_Custom, 
            'No_EFA_MonthSinCos_CUT': Dataset_Custom,
            'No_EFA_Day_MonthSinCos_CUT': Dataset_Custom,
            'July_Important_Variable_CUT': Dataset_Custom,
            'July_High_Correlation_Table_CUT': Dataset_Custom,
            'ALL_July': Dataset_Custom,
            'No_MonthSinCos_July': Dataset_Custom, 
            'No_EFA_MonthSinCos_July': Dataset_Custom,
            'No_EFA_Day_MonthSinCos_July': Dataset_Custom,
            'July_Important_Variable_July': Dataset_Custom,
            'July_High_Correlation_Table_July': Dataset_Custom,
            'ALL_CUT_July': Dataset_Custom,
            'No_MonthSinCos_CUT_July': Dataset_Custom, 
            'No_EFA_MonthSinCos_CUT_July': Dataset_Custom,
            'No_EFA_Day_MonthSinCos_CUT_July': Dataset_Custom,
            'July_Important_Variable_CUT_July': Dataset_Custom,
            'July_High_Correlation_Table_CUT_July': Dataset_Custom,
            'ALL_TEMP': Dataset_Custom,
            'July_High_Correlation_Table_TEMP': Dataset_Custom,
            'ALL_CUT_TEMP': Dataset_Custom,
            'July_High_Correlation_Table_CUT_TEMP': Dataset_Custom,
            'ALL_TEMP_WIND': Dataset_Custom,
            'July_High_Correlation_Table_TEMP_WIND': Dataset_Custom,
            'ALL_CUT_TEMP_WIND': Dataset_Custom,
            'July_High_Correlation_Table_CUT_TEMP_WIND': Dataset_Custom,
            'August_High_Correlation_TEMP_WIND': Dataset_Custom, 
            'August_High_Correlation_TEMP': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set)) 
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return model_optim
    
    def _select_criterion(self, losstype):
        if losstype == "mse":
            criterion = nn.MSELoss()
        elif losstype == "mae":
            criterion = nn.L1Loss()
        elif losstype == "UncertaintyLoss":
            print("Custom loss function1")
            criterion = UncertaintyLoss
        else:
            criterion = nn.L1Loss()
        return criterion

    def valid(self, valid_data, valid_loader, criterion):
        self.model.eval()
        total_loss = []

        preds = []
        trues = []
        mids = []
        pred_scales = []
        true_scales = []
        mid_scales = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(valid_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                valid_data, batch_x, batch_y)

            if self.args.stacks == 1:
                loss = criterion(pred.detach().cpu(), true.detach().cpu())

                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                pred_scales.append(pred_scale.detach().cpu().numpy())
                true_scales.append(true_scale.detach().cpu().numpy())

            elif self.args.stacks == 2:
                loss = criterion(pred.detach().cpu(), true.detach().cpu()) + criterion(mid.detach().cpu(), true.detach().cpu())

                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                mids.append(mid.detach().cpu().numpy())
                pred_scales.append(pred_scale.detach().cpu().numpy())
                mid_scales.append(mid_scale.detach().cpu().numpy())
                true_scales.append(true_scale.detach().cpu().numpy())

            else:
                print('Error!')

            total_loss.append(loss)
        total_loss = np.average(total_loss)

        if self.args.stacks == 1:
            preds = np.array(preds)
            trues = np.array(trues)
            pred_scales = np.array(pred_scales)
            true_scales = np.array(true_scales)

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
            pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])

            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
            print('normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
            print('denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))
        elif self.args.stacks == 2:
            preds = np.array(preds)
            trues = np.array(trues)
            mids = np.array(mids)
            pred_scales = np.array(pred_scales)
            true_scales = np.array(true_scales)
            mid_scales = np.array(mid_scales)

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            mids = mids.reshape(-1, mids.shape[-2], mids.shape[-1])
            true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
            pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])
            mid_scales = mid_scales.reshape(-1, mid_scales.shape[-2], mid_scales.shape[-1])
            # print('test shape:', preds.shape, mids.shape, trues.shape)

            mae, mse, rmse, mape, mspe, corr = metric(mids, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(mid_scales, true_scales)
            print('mid --> normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
            print('mid --> denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))

            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
            print('final --> normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
            print('final --> denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))
        else:
            print('Error!')

        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        valid_data, valid_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')
        path = os.path.join(self.args.checkpoints, setting)
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        writer = SummaryWriter('event/run_ETTh/{}'.format(self.args.model_name))

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if self.args.resume:
            self.model, lr, epoch_start = load_model(self.model, path, model_name=self.args.data, horizon=self.args.horizon)
        else:
            epoch_start = 0

        for epoch in range(epoch_start, self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                    train_data, batch_x, batch_y)

                if self.args.stacks == 1:
                    loss = criterion(pred, true)
                elif self.args.stacks == 2:
                    loss = criterion(pred, true) + criterion(mid, true)
                else:
                    print('Error!')

                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    print('use amp')    
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            print('--------start to validate-----------')
            valid_loss = self.valid(valid_data, valid_loader, criterion)
            print('--------start to test-----------')
            test_loss = self.valid(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} valid Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, valid_loss, test_loss))

            writer.add_scalar('train_loss', train_loss, global_step=epoch)
            writer.add_scalar('valid_loss', valid_loss, global_step=epoch)
            writer.add_scalar('test_loss', test_loss, global_step=epoch)

            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            lr = adjust_learning_rate(model_optim, epoch+1, self.args)
            
        save_model(epoch, lr, self.model, path, model_name=self.args.data, horizon=self.args.pred_len)
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, evaluate=False):
        test_data, test_loader = self._get_data(flag='test')
        print(f"len of test in the function test = {len(test_data)}")
        self.model.eval()
        
        preds = []
        trues = []
        mids = []
        pred_scales = []
        true_scales = []
        mid_scales = []
        
        if evaluate:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                test_data, batch_x, batch_y)

            if self.args.stacks == 1:
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                pred_scales.append(pred_scale.detach().cpu().numpy())
                true_scales.append(true_scale.detach().cpu().numpy())
            elif self.args.stacks == 2:
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                mids.append(mid.detach().cpu().numpy())
                pred_scales.append(pred_scale.detach().cpu().numpy())
                mid_scales.append(mid_scale.detach().cpu().numpy())
                true_scales.append(true_scale.detach().cpu().numpy())

            else:
                print('Error!')

        if self.args.stacks == 1:
            preds = np.array(preds)
            trues = np.array(trues)

            pred_scales = np.array(pred_scales)
            true_scales = np.array(true_scales)

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
            pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])

            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
            print('normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
            print('TTTT denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))

        elif self.args.stacks == 2:
            preds = np.array(preds)
            trues = np.array(trues)
            mids = np.array(mids)

            pred_scales = np.array(pred_scales)
            true_scales = np.array(true_scales)
            mid_scales = np.array(mid_scales)

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            mids = mids.reshape(-1, mids.shape[-2], mids.shape[-1])
            true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
            pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])
            mid_scales = mid_scales.reshape(-1, mid_scales.shape[-2], mid_scales.shape[-1])
            # print('test shape:', preds.shape, mids.shape, trues.shape)

            mae, mse, rmse, mape, mspe, corr = metric(mids, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(mid_scales, true_scales)
            print('Mid --> normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))

            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
            print('TTTT Final --> denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))

        else:
            print('Error!')

        # result save
        if self.args.save:
            folder_path = 'exp/ett_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            print('Test:mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))

            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
            np.save(folder_path + 'pred_scales.npy', pred_scales)
            np.save(folder_path + 'true_scales.npy', true_scales)
            
        return mae, maes, mse, mses

    def _process_one_batch_SCINet(self, dataset_object, batch_x, batch_y):
        #batch_x = batch_x.double().cuda()
        batch_x = batch_x.double()
        batch_y = batch_y.double()

        if self.args.stacks == 1:
            outputs = self.model(batch_x)
        elif self.args.stacks == 2:
            outputs, mid = self.model(batch_x)
        else:
            print('Error!')

        #if self.args.inverse:
        outputs_scaled = dataset_object.inverse_transform(outputs)
        if self.args.stacks == 2:
            mid_scaled = dataset_object.inverse_transform(mid)
        f_dim = -1 if self.args.features=='MS' else 0
        #batch_y = batch_y[:,-self.args.pred_len:,f_dim:].cuda()
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:]
        batch_y_scaled = dataset_object.inverse_transform(batch_y)

        if self.args.stacks == 1:
            return outputs, outputs_scaled, 0,0, batch_y, batch_y_scaled
        elif self.args.stacks == 2:
            return outputs, outputs_scaled, mid, mid_scaled, batch_y, batch_y_scaled
        else:
            print('Error!')
