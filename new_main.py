# TODO:
# Correct for the number of label classes.

import torch.optim as optim
import torch.utils.data as util_data
import itertools
import torch
from new_dataloader import image_Loader
from new_preprocessing import image_trian, landmark_transform
import new_model
import new_lr_schedule
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import math 
import torch.nn as nn
from PIL import Image
import pandas as pd

def str2bool(v):
    return v.lower() in ('true')

def tensor2img(img):
    img = img.data.numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    img = (np.transpose(img, (1, 2, 0))+ 1) / 2.0 * 255.0
    return img.astype(np.uint8)

def save_img(img, name, path):
    img = tensor2img(img)
    img = Image.fromarray(img)
    img.save(path + name + '.png')
    return img


def AU_detection_evalv1(loader, region_learning, align_net, local_attention_refine,
                local_au_net, global_au_feat, au_net, use_gpu=True, fail_threshold=0.1):
    missing_label = 9
    for i, batch in enumerate(loader):
        input, land, biocular, au = batch
        if use_gpu:
            input, land, au = input.cuda(), land.cuda(), au.cuda()

        region_feat = region_learning(input)
        align_feat, align_output, aus_map = align_net(region_feat)
        if use_gpu:
            aus_map = aus_map.cuda()
        output_aus_map = local_attention_refine(aus_map.detach())
        local_au_out_feat = local_au_net(region_feat, output_aus_map)
        global_au_out_feat = global_au_feat(region_feat)
        concat_au_feat = torch.cat((align_feat, global_au_out_feat, local_au_out_feat), 1)
        aus_output = au_net(concat_au_feat)
        aus_output = (aus_output[:,1,:]).exp()

        if i == 0:
            all_output = aus_output.data.cpu().float()
            all_au = au.data.cpu().float()
            all_pred_land = align_output.data.cpu().float()
            all_land = land.data.cpu().float()
        else:
            all_output = torch.cat((all_output, aus_output.data.cpu().float()), 0)
            all_au = torch.cat((all_au, au.data.cpu().float()), 0)
            all_pred_land = torch.cat((all_pred_land, align_output.data.cpu().float()), 0)
            all_land = torch.cat((all_land, land.data.cpu().float()), 0)

    AUoccur_pred_prob = all_output.data.numpy()
    AUoccur_actual = all_au.data.numpy()
    pred_land = all_pred_land.data.numpy()
    GT_land = all_land.data.numpy()

    # AUs
    AUoccur_pred = np.zeros(AUoccur_pred_prob.shape)
    AUoccur_pred[AUoccur_pred_prob < 0.5] = 0
    AUoccur_pred[AUoccur_pred_prob >= 0.5] = 1

    AUoccur_actual = AUoccur_actual.transpose((1, 0))
    AUoccur_pred = AUoccur_pred.transpose((1, 0))

    f1score_arr = np.zeros(AUoccur_actual.shape[0])
    acc_arr = np.zeros(AUoccur_actual.shape[0])
    for i in range(AUoccur_actual.shape[0]):
        curr_actual = AUoccur_actual[i]
        curr_pred = AUoccur_pred[i]

        new_curr_actual = curr_actual[curr_actual != missing_label]
        new_curr_pred = curr_pred[curr_actual != missing_label]

        f1score_arr[i] = f1_score(new_curr_actual, new_curr_pred)
        acc_arr[i] = accuracy_score(new_curr_actual, new_curr_pred)

    # landmarks
    errors = np.zeros((GT_land.shape[0], int(GT_land.shape[1] / 2)))
    mean_errors = np.zeros(GT_land.shape[0])
    for i in range(GT_land.shape[0]):
        left_eye_x = GT_land[i, (20 - 1) * 2:(26 - 1) * 2:2]
        l_ocular_x = left_eye_x.mean()
        left_eye_y = GT_land[i, (20 - 1) * 2 + 1:(26 - 1) * 2 + 1:2]
        l_ocular_y = left_eye_y.mean()

        right_eye_x = GT_land[i, (26 - 1) * 2:(32 - 1) * 2:2]
        r_ocular_x = right_eye_x.mean()
        right_eye_y = GT_land[i, (26 - 1) * 2 + 1:(32 - 1) * 2 + 1:2]
        r_ocular_y = right_eye_y.mean()

        biocular = math.sqrt((l_ocular_x - r_ocular_x) ** 2 + (l_ocular_y - r_ocular_y) ** 2)

        for j in range(0, GT_land.shape[1], 2):
            errors[i, int(j / 2)] = math.sqrt((GT_land[i, j] - pred_land[i, j]) ** 2 + (
                    GT_land[i, j + 1] - pred_land[i, j + 1]) ** 2) / biocular

        mean_errors[i] = errors[i].mean()
    mean_error = mean_errors.mean()

    failure_ind = np.zeros(len(GT_land))
    failure_ind[mean_errors > fail_threshold] = 1
    failure_rate = failure_ind.sum() / failure_ind.shape[0]

    return f1score_arr, acc_arr, mean_error, failure_rate


def AU_detection_evalv2(loader, region_learning, align_net, local_attention_refine,
                local_au_net, global_au_feat, au_net, use_gpu=True, fail_threshold = 0.1):
    missing_label = 9
    for i, batch in enumerate(loader):
        input, land, biocular, au  = batch
        if use_gpu:
            input, land, au = input.cuda(), land.cuda(), au.cuda()

        region_feat = region_learning(input)
        align_feat, align_output, aus_map = align_net(region_feat)
        if use_gpu:
            aus_map = aus_map.cuda()
        output_aus_map = local_attention_refine(aus_map.detach())
        local_au_out_feat, local_aus_output = local_au_net(region_feat, output_aus_map)
        local_aus_output = (local_aus_output[:, 1, :]).exp()
        global_au_out_feat = global_au_feat(region_feat)
        concat_au_feat = torch.cat((align_feat, global_au_out_feat, local_au_out_feat.detach()), 1)
        aus_output = au_net(concat_au_feat)
        aus_output = (aus_output[:,1,:]).exp()

        if i == 0:
            all_local_output = local_aus_output.data.cpu().float()
            all_output = aus_output.data.cpu().float()
            all_au = au.data.cpu().float()
            all_pred_land = align_output.data.cpu().float()
            all_land = land.data.cpu().float()
        else:
            all_local_output = torch.cat((all_local_output, local_aus_output.data.cpu().float()), 0)
            all_output = torch.cat((all_output, aus_output.data.cpu().float()), 0)
            all_au = torch.cat((all_au, au.data.cpu().float()), 0)
            all_pred_land = torch.cat((all_pred_land, align_output.data.cpu().float()), 0)
            all_land = torch.cat((all_land, land.data.cpu().float()), 0)

    AUoccur_pred_prob = all_output.data.numpy()
    local_AUoccur_pred_prob = all_local_output.data.numpy()
    AUoccur_actual = all_au.data.numpy()
    pred_land = all_pred_land.data.numpy()
    GT_land = all_land.data.numpy()
    # np.savetxt('BP4D_part1_pred_land_49.txt', pred_land, fmt='%.4f', delimiter='\t')
    np.savetxt('B3D_val_predAUprob-2_all_.txt', AUoccur_pred_prob, fmt='%f',
               delimiter='\t')
    # AUs
    AUoccur_pred = np.zeros(AUoccur_pred_prob.shape)
    AUoccur_pred[AUoccur_pred_prob < 0.5] = 0
    AUoccur_pred[AUoccur_pred_prob >= 0.5] = 1
    local_AUoccur_pred = np.zeros(local_AUoccur_pred_prob.shape)
    local_AUoccur_pred[local_AUoccur_pred_prob < 0.5] = 0
    local_AUoccur_pred[local_AUoccur_pred_prob >= 0.5] = 1

    AUoccur_actual = AUoccur_actual.transpose((1, 0))
    AUoccur_pred = AUoccur_pred.transpose((1, 0))
    local_AUoccur_pred = local_AUoccur_pred.transpose((1, 0))

    f1score_arr = np.zeros(AUoccur_actual.shape[0])
    acc_arr = np.zeros(AUoccur_actual.shape[0])
    local_f1score_arr = np.zeros(AUoccur_actual.shape[0])
    local_acc_arr = np.zeros(AUoccur_actual.shape[0])
    for i in range(AUoccur_actual.shape[0]):
        curr_actual = AUoccur_actual[i]
        curr_pred = AUoccur_pred[i]
        local_curr_pred = local_AUoccur_pred[i]

        new_curr_actual = curr_actual[curr_actual != missing_label]
        new_curr_pred = curr_pred[curr_actual != missing_label]
        local_new_curr_pred = local_curr_pred[curr_actual != missing_label]

        f1score_arr[i] = f1_score(new_curr_actual, new_curr_pred)
        acc_arr[i] = accuracy_score(new_curr_actual, new_curr_pred)
        local_f1score_arr[i] = f1_score(new_curr_actual, local_new_curr_pred)
        local_acc_arr[i] = accuracy_score(new_curr_actual, local_new_curr_pred)

    # landmarks
    errors = np.zeros((GT_land.shape[0], int(GT_land.shape[1] / 2)))
    mean_errors = np.zeros(GT_land.shape[0])
    for i in range(GT_land.shape[0]):
        left_eye_x = GT_land[i, (20 - 1) * 2:(26 - 1) * 2:2]
        l_ocular_x = left_eye_x.mean()
        left_eye_y = GT_land[i, (20 - 1) * 2 + 1:(26 - 1) * 2 + 1:2]
        l_ocular_y = left_eye_y.mean()

        right_eye_x = GT_land[i, (26 - 1) * 2:(32 - 1) * 2:2]
        r_ocular_x = right_eye_x.mean()
        right_eye_y = GT_land[i, (26 - 1) * 2 + 1:(32 - 1) * 2 + 1:2]
        r_ocular_y = right_eye_y.mean()

        biocular = math.sqrt((l_ocular_x - r_ocular_x) ** 2 + (l_ocular_y - r_ocular_y) ** 2)

        for j in range(0, GT_land.shape[1], 2):
            errors[i, int(j / 2)] = math.sqrt((GT_land[i, j] - pred_land[i, j]) ** 2 + (
                    GT_land[i, j + 1] - pred_land[i, j + 1]) ** 2) / biocular

        mean_errors[i] = errors[i].mean()
    mean_error = mean_errors.mean()

    failure_ind = np.zeros(len(GT_land))
    failure_ind[mean_errors > fail_threshold] = 1
    failure_rate = failure_ind.sum() / failure_ind.shape[0]

    return local_f1score_arr, local_acc_arr, f1score_arr, acc_arr, mean_error, failure_rate


def vis_attention(loader, region_learning, align_net, local_attention_refine, write_path_prefix, net_name, epoch, alpha = 0.5, use_gpu=True):
    for i, batch in enumerate(loader):
        input, land, biocular, au = batch
        # if i > 1:
        #     break
        if use_gpu:
            input = input.cuda()
        region_feat = region_learning(input)
        align_feat, align_output, aus_map = align_net(region_feat)
        if use_gpu:
            aus_map = aus_map.cuda()
        output_aus_map = local_attention_refine(aus_map.detach())

        # aus_map is predefined, and output_aus_map is refined
        spatial_attention = output_aus_map #aus_map
        if i == 0:
            all_input = input.data.cpu().float()
            all_spatial_attention = spatial_attention.data.cpu().float()
        else:
            all_input = torch.cat((all_input, input.data.cpu().float()), 0)
            all_spatial_attention = torch.cat((all_spatial_attention, spatial_attention.data.cpu().float()), 0)

    for i in range(all_spatial_attention.shape[0]):
        background = save_img(all_input[i], 'input', write_path_prefix + net_name + '/vis_map/' + str(epoch) +
                        '/' + str(i) + '_')
        for j in range(all_spatial_attention.shape[1]):
            fig, ax = plt.subplots()
            # print(all_spatial_attention[i,j].max(), all_spatial_attention[i,j].min())
            # cax = ax.imshow(all_spatial_attention[i,j], cmap='jet', interpolation='bicubic')
            cax = ax.imshow(all_spatial_attention[i, j], cmap='jet', interpolation='bicubic', vmin=0, vmax=1)
            ax.axis('off')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            #        cbar = fig.colorbar(cax)
            fig.savefig(write_path_prefix + net_name + '/vis_map/' + str(epoch) +
                        '/' + str(i) + '_au_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)

        for j in range(all_spatial_attention.shape[1]):
            overlay = Image.open(write_path_prefix + net_name + '/vis_map/' + str(epoch) +
                        '/' + str(i) + '_au_' + str(j) + '.png')
            overlay = overlay.resize(background.size, Image.ANTIALIAS)
            background = background.convert('RGBA')
            overlay = overlay.convert('RGBA')
            new_img = Image.blend(background, overlay, alpha)
            new_img.save(write_path_prefix + net_name + '/overlay_vis_map/' + str(epoch) +
                        '/' + str(i) + '_au_' + str(j) + '.png', 'PNG')


def dice_loss(pred, target, smooth = 1):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth)) / iflat.size(0)


def au_softmax_loss(input, target, weight=None, size_average=True, reduce=True):
    classify_loss = nn.NLLLoss(size_average=size_average, reduce=reduce)

    for i in range(input.size(2)):
        t_input = input[:, :, i]
        t_target = target[:, i]
        t_loss = classify_loss(t_input, t_target)
        if weight is not None:
            t_loss = t_loss * weight[i]
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def au_dice_loss(input, target, weight=None, smooth = 1, size_average=True):
    for i in range(input.size(2)):
        # input is log_softmax, t_input is probability
        t_input = (input[:, 1, i]).exp()
        t_target = (target[:, i]).float()
        # t_loss = 1 - float(2*torch.dot(t_input, t_target) + smooth)/\
        #          (torch.dot(t_input, t_input)+torch.dot(t_target, t_target)+smooth)/t_input.size(0)
        t_loss = dice_loss(t_input, t_target, smooth)
        if weight is not None:
            t_loss = t_loss * weight[i]
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def landmark_loss(input, target, biocular, size_average=True):
    for i in range(input.size(0)):
        t_input = input[i,:]
        t_target = target[i,:]
        t_loss = torch.sum((t_input - t_target) ** 2) / (2.0*biocular[i])
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def attention_refine_loss(input, target, size_average=True, reduce=True):
    # loss is averaged over each point in the attention map,
    # note that Eq.(4) in our ECCV paper is to sum all the points,
    # change the value of lambda_refine can remove this difference.
    classify_loss = nn.BCELoss(size_average=size_average, reduce=reduce)

    input = input.view(input.size(0), input.size(1), -1)
    target = target.view(target.size(0), target.size(1), -1)
    for i in range(input.size(1)):
        t_input = input[:, i, :]
        t_target = target[:, i, :]
        t_loss = classify_loss(t_input, t_target)
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)
    # sum losses of all AUs
    return loss.sum()

def calculate_AU_weight(occurence_df):
        """
        Calculates the AU weight according to a occurence dataframe 
        inputs: 
            occurence_df: a pandas dataframe containing occurence of each AU. See BP4D+
        """
        #occurence_df = occurence_df.rename(columns = {'two':'new_name'})
        #occurence_df2 = occurence_df.iloc[:,2::]
        occurence_df2 = occurence_df[['1','2', '4','6','7','10','12','14','15','17','23','24']]
        weight_mtrx = np.zeros((occurence_df2.shape[1], 1))
        for i in range(occurence_df2.shape[1]):
            weight_mtrx[i] = np.sum(occurence_df2.iloc[:, i]
                                    > 0) / float(occurence_df2.shape[0])
        weight_mtrx = 1.0/weight_mtrx

        #print(weight_mtrx)
        weight_mtrx[weight_mtrx == np.inf] = 0
        #print(np.sum(weight_mtrx)*len(weight_mtrx))
        weight_mtrx = weight_mtrx / (np.sum(weight_mtrx)*len(weight_mtrx))

        return(weight_mtrx)


def main():

    # def train_val_dataset(dataset,val_split = 0.25):
    #     train_idx,val_idx = train_test_split(list(range(len(dataset))),test_size = 0.25)
    #     datasets = {}
    #     datasets['train'] = Subset(dataset,train_idx)
    #     datasets['test'] = Subset(dataset,val_idx)
    #     return(datasets)

    use_gpu = torch.cuda.is_available()

    
    dsets = {}
    dset_loaders = {}

    dsets = image_Loader(crop_size = CONFIG_CROP_SIZE, csv_dir = CONFIG_csvdir, img_dir = CONFIG_imgdir,
                                    transform = image_trian(crop_size=CONFIG_CROP_SIZE), phase = 'train',
                                    target_transform=landmark_transform(img_size=CONFIG_CROP_SIZE, flip_reflect=np.loadtxt(CONFIG_flip_reflect,dtype = 'int',delimiter=","))
                        )
    train_set,test_set = torch.utils.data.random_split(dsets,[6000,531])
    

    # CHANGEME
    au_mk0 = pd.read_csv(CONFIG_csvdir)
    au_mk0 = calculate_AU_weight(au_mk0[['1','2', '4','6','7','10','12','14','15','17','23','24']])
    au_weight = torch.from_numpy(au_mk0)

    if use_gpu:
        au_weight = au_weight.float().cuda()
    else:
        au_weight = au_weight.float()


    train_loader = util_data.DataLoader(dataset=train_set,batch_size=8,shuffle=True)
    test_loader = util_data.DataLoader(dataset=test_set,batch_size=8,shuffle=True)
    
    dset_loaders['train'] = train_loader
    dset_loaders['test'] = test_loader


# network_dict = {'HLFeatExtractor':HLFeatExtractor, 'HMRegionLearning':HMRegionLearning,
#                 'AlignNet':AlignNet, 'LocalAttentionRefine':LocalAttentionRefine,
#                 'LocalAUNetv2':LocalAUNetv2, 'AUNet':AUNet
#                 }


    # Set network modules:
    region_learning = new_model.network_dict['HMRegionLearning'](input_dim=3, unit_dim=CONFIG_unit_dim)
    
    align_net = new_model.network_dict['AlignNet'](crop_size=CONFIG_CROP_SIZE, map_size=CONFIG_map_size, au_num = CONFIG_au_num,
                                                        land_num=CONFIG_land_num, input_dim = CONFIG_unit_dim*8,fill_coeff = CONFIG_fill_coeff)
    local_attention_refine = new_model.network_dict['LocalAttentionRefine'](au_num = CONFIG_au_num, unit_dim=CONFIG_unit_dim)
    
    local_au_net = new_model.network_dict['LocalAUNetv2'](au_num=CONFIG_au_num, input_dim=CONFIG_unit_dim*8,
                                                                                     unit_dim=CONFIG_unit_dim)

    global_au_feat = new_model.network_dict['HLFeatExtractor'](input_dim=CONFIG_unit_dim*8,
                                                                                     unit_dim=CONFIG_unit_dim)
    au_net = new_model.network_dict['AUNet'](au_num=CONFIG_au_num, input_dim = 12000, unit_dim = CONFIG_unit_dim)

    #network_dict[config.region_learning]()

    #dset_loaders['train'] = util_data.DataLoader(dsets['train'], batch_size=config.train_batch_size, shuffle = True, num_workers=config.num_worker)

    #dsets['test'] = image_Loader(crop_size = CONFIG_CROP_SIZE, csv_dir = CONFIG_csvdir, img_dir = CONFIG_imgdir,
    #                                transform = None, target_transform=None, phase = 'test')
    if use_gpu:
        region_learning = region_learning.cuda()
        align_net = align_net.cuda()
        local_attention_refine = local_attention_refine.cuda()
        local_au_net = local_au_net.cuda()
        global_au_feat = global_au_feat.cuda()
        au_net = au_net.cuda()
    
    print(region_learning)
    print(align_net)
    print(local_attention_refine)
    print(local_au_net)
    print(global_au_feat)
    print(au_net)

    optim_dict = {'SGD': optim.SGD, 'Adam': optim.Adam}

    region_learning_parameter_list = [{'params': filter(lambda p: p.requires_grad, region_learning.parameters()), 'lr': 1}]
    align_net_parameter_list = [
        {'params': filter(lambda p: p.requires_grad, align_net.parameters()), 'lr': 1}]
    local_attention_refine_parameter_list = [
        {'params': filter(lambda p: p.requires_grad, local_attention_refine.parameters()), 'lr': 1}]
    local_au_net_parameter_list = [
        {'params': filter(lambda p: p.requires_grad, local_au_net.parameters()), 'lr': 1}]
    global_au_feat_parameter_list = [
        {'params': filter(lambda p: p.requires_grad, global_au_feat.parameters()), 'lr': 1}]
    au_net_parameter_list = [
        {'params': filter(lambda p: p.requires_grad, au_net.parameters()), 'lr': 1}]

    optimizer = optim_dict[CONFIG_optimizer_type](itertools.chain(region_learning_parameter_list, align_net_parameter_list,
                                                                local_attention_refine_parameter_list,
                                                                local_au_net_parameter_list,
                                                                global_au_feat_parameter_list,
                                                                au_net_parameter_list),
                                                lr=1.0, momentum=CONFIG_momentum, weight_decay=CONFIG_weight_decay,
                                                nesterov=CONFIG_use_nesterov)

    param_lr = []

    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])
    
    lr_scheduler = new_lr_schedule.schedule_dict[CONFIG_lr_type]
    res_file = open(
        CONFIG_write_res_prefix + CONFIG_run_name + '/AU_pred_' + str(CONFIG_start_epoch) + '.txt', 'w')

    ## train
    count = 0

    for epoch in range(CONFIG_start_epoch, CONFIG_n_epochs + 1):
        if epoch > CONFIG_start_epoch:
            print('taking snapshot ...')
            torch.save(region_learning.state_dict(),
                       CONFIG_write_path_prefix + CONFIG_run_name + '/region_learning_' + str(epoch) + '.pth')
            torch.save(align_net.state_dict(),
                       CONFIG_write_path_prefix + CONFIG_run_name + '/align_net_' + str(epoch) + '.pth')
            torch.save(local_attention_refine.state_dict(),
                       CONFIG_write_path_prefix + CONFIG_run_name + '/local_attention_refine_' + str(epoch) + '.pth')
            torch.save(local_au_net.state_dict(),
                       CONFIG_write_path_prefix + CONFIG_run_name + '/local_au_net_' + str(epoch) + '.pth')
            torch.save(global_au_feat.state_dict(),
                       CONFIG_write_path_prefix + CONFIG_run_name + '/global_au_feat_' + str(epoch) + '.pth')
            torch.save(au_net.state_dict(),
                       CONFIG_write_path_prefix + CONFIG_run_name + '/au_net_' + str(epoch) + '.pth')

        # eval in the train
        if epoch > CONFIG_start_epoch:
            print('testing ...')
            region_learning.train(False)
            align_net.train(False)
            local_attention_refine.train(False)
            local_au_net.train(False)
            global_au_feat.train(False)
            au_net.train(False)

            local_f1score_arr, local_acc_arr, f1score_arr, acc_arr, mean_error, failure_rate = AU_detection_evalv2(
                dset_loaders['test'], region_learning, align_net, local_attention_refine,
                local_au_net, global_au_feat, au_net, use_gpu=use_gpu)
            print('epoch =%d, local f1 score mean=%f, local accuracy mean=%f, '
                  'f1 score mean=%f, accuracy mean=%f, mean error=%f, failure rate=%f' % (epoch, local_f1score_arr.mean(),
                                local_acc_arr.mean(), f1score_arr.mean(),
                                acc_arr.mean(), mean_error, failure_rate))
            print('%d\t%f\t%f\t%f\t%f\t%f\t%f' % (epoch, local_f1score_arr.mean(),
                                                local_acc_arr.mean(), f1score_arr.mean(),
                                                acc_arr.mean(), mean_error, failure_rate), file=res_file)

            region_learning.train(True)
            align_net.train(True)
            local_attention_refine.train(True)
            local_au_net.train(True)
            global_au_feat.train(True)
            au_net.train(True)

        if epoch >= CONFIG_n_epochs:
            break

        for i, batch in enumerate(dset_loaders['train']):
            if i % CONFIG_display == 0 and count > 0:
                print('[epoch = %d][iter = %d][total_loss = %f][loss_au_softmax = %f][loss_au_dice = %f]'
                      '[loss_local_au_softmax = %f][loss_local_au_dice = %f]'
                      '[loss_land = %f]' % (epoch, i,
                    total_loss.data.cpu().numpy(), loss_au_softmax.data.cpu().numpy(), loss_au_dice.data.cpu().numpy(),
                    loss_local_au_softmax.data.cpu().numpy(), loss_local_au_dice.data.cpu().numpy(), loss_land.data.cpu().numpy()))
                print('learning rate = %f %f %f %f %f %f' % (optimizer.param_groups[0]['lr'],
                                                          optimizer.param_groups[1]['lr'],
                                                          optimizer.param_groups[2]['lr'],
                                                          optimizer.param_groups[3]['lr'],
                                                          optimizer.param_groups[4]['lr'],
                                                          optimizer.param_groups[5]['lr']))
                print('the number of training iterations is %d' % (count))

            input, land, biocular, au = batch

            if use_gpu:
                input, land, biocular, au = input.cuda(), land.float().cuda(), \
                                            biocular.float().cuda(), au.long().cuda()
            else:
                au = au.long()

            optimizer = lr_scheduler(param_lr, optimizer, epoch, CONFIG_gamma, CONFIG_stepsize, CONFIG_init_lr)
            optimizer.zero_grad()

            region_feat = region_learning(input)
            align_feat, align_output, aus_map = align_net(region_feat)
            if use_gpu:
                aus_map = aus_map.cuda()
            output_aus_map = local_attention_refine(aus_map.detach())
            local_au_out_feat, local_aus_output = local_au_net(region_feat, output_aus_map)
            global_au_out_feat = global_au_feat(region_feat)
            concat_au_feat = torch.cat((align_feat, global_au_out_feat, local_au_out_feat.detach()), 1)
            aus_output = au_net(concat_au_feat)

            loss_au_softmax = au_softmax_loss(aus_output, au, weight=au_weight)
            loss_au_dice = au_dice_loss(aus_output, au, weight=au_weight)
            loss_au = loss_au_softmax + loss_au_dice

            loss_local_au_softmax = au_softmax_loss(local_aus_output, au, weight=au_weight)
            loss_local_au_dice = au_dice_loss(local_aus_output, au, weight=au_weight)
            loss_local_au = loss_local_au_softmax + loss_local_au_dice

            loss_land = landmark_loss(align_output, land, biocular)

            total_loss = CONFIG_lambda_au * (loss_au + loss_local_au) + \
                         CONFIG_lambda_land * loss_land

            total_loss.backward()
            optimizer.step()

            count = count + 1

    res_file.close()

CONFIG_CROP_SIZE = 176
CONFIG_csvdir = "F:\\here.csv"
CONFIG_imgdir = "F:\\FaceExprDecode\\"
CONFIG_flip_reflect = "F:\\facial_expression\\reflect_68.txt"
CONFIG_unit_dim = 8
CONFIG_land_num = 68
CONFIG_fill_coeff = 0.56
CONFIG_map_size = 44
CONFIG_au_num = 12
CONFIG_optimizer_type = 'SGD'
CONFIG_momentum = 0.9
CONFIG_weight_decay = 0.0005
CONFIG_use_nesterov = True
CONFIG_lr_type = 'step'
CONFIG_start_epoch = 0
CONFIG_n_epochs = 10
CONFIG_display = 100
CONFIG_gamma = 0.3
CONFIG_stepsize = 2
CONFIG_init_lr = 0.01
CONFIG_lambda_au = 1
CONFIG_lambda_land = 0.5
CONFIG_write_res_prefix = "F:/facial_expression/res/"
CONFIG_write_path_prefix = "F:/facial_expression/snapshots/"
CONFIG_run_name = "JAAV2"

main()

# %%
