#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 @Author: Shaobo Yang
 @Time:12/12/2019 20:10 PM
 @Email: yang0123@mail.ustc.edu.cn
"""

import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from reader.reader import Reader
from train_cnn import CNN
from config.config import Config
from plot.plot_test import plot_test
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Pick_v(object):
    """ Pick the dispersion curves.

    Attributes:

    """

    def __init__(self):
        self.reader = Reader()
        self.config = Config()
        self.model = CNN()
        self.sess = tf.Session()
        

    def pick_v(self):
        """ Pick the dispersion curves using the trained CNN.

        """
        # Load station info.
        config = self.config
        auto_refT = not bool(config.ref_T)
        station = {}
        sum_test_loss = 0
        with open('./config/station.txt') as f:
            for each_line in f.readlines():
                item = each_line.split()
                station[item[0]] = [float(item[1]), float(item[2])]

        print('Start picking!')
        os.chdir(config.result_path + '/test_result/')

        # Load the trained model.
        saver = tf.train.Saver()
        ckpt = config.root + '/saver/-10000'
        saver.restore(self.sess, ckpt)
        print('Restored CNN model from checkpoint' + ckpt)

        file_list = self.reader.get_test_file()
        file_num = len(file_list)
        batch_list = []        # Split 'file_list' into 'batch_list'. 
        batch_num = math.ceil(file_num/config.batch_size)
        for i in range(batch_num - 1):
            b = i*config.batch_size
            e = (i + 1)*config.batch_size
            batch_list.append(file_list[b:e])
        batch_list.append(file_list[file_num - config.batch_size:])

        # Extract the dispersion curves for each batch.
        for k in range(batch_num):
            batch_input = []
            n = 0
            batch_true_vG = []
            batch_true_vC = []
            batch_label = []
            for file in batch_list[k]:
                n += 1
                print(n + k*config.batch_size, file)
                file_path_dispG = (config.test_data_path + '/group_image/' + file + '.dat')
                each_dispG = self.reader.get_disp_matrix(file_path_dispG, config.input_size[:2])
                each_disp_dispC = (config.test_data_path + '/phase_image/' + file + '.dat')
                each_dispC = self.reader.get_disp_matrix(each_disp_dispC, config.input_size[:2])
                each_data = [each_dispG, each_dispC]
                batch_input.append(each_data)

                if config.test:
                    file_path_vG = (config.test_data_path + '/group_velocity/' + file + '.txt')
                    file_path_vC = (config.test_data_path + '/phase_velocity/' + file + '.txt')
                    try:
                        true_vG = np.loadtxt(file_path_vG)       
                    except:
                        true_vG = np.zeros([config.input_size[1], 2])
                    batch_true_vG.append(true_vG[:config.input_size[1], 1])

                    try:
                        true_vC = np.loadtxt(file_path_vC)
                    except:
                        true_vC = np.zeros([config.input_size[1],2])
                    batch_true_vC.append(true_vC[:config.input_size[1], 1])

                    true_probG = self.reader.get_label_matrix(file_path_vG, config.input_size[:2])
                    true_probC = self.reader.get_label_matrix(file_path_vC, config.input_size[:2])
                    each_label = [true_probG, true_probC]
                    batch_label.append(each_label)
                
            batch_input = np.array(batch_input)                   # [file, channel, V, T]
            batch_input = batch_input.transpose((0, 2, 3, 1))     # [file, V, T, channel]

            # batch_pred_prob: Predicted probability maps.
            batch_pred_prob = self.model.predict(sess=self.sess, input=batch_input)    

            # transpose: [file, V, T, channel] to [channel, file, V, T]
            batch_pred_prob = batch_pred_prob.transpose((3, 0, 1, 2))   
            batch_pred_probG = batch_pred_prob[0]
            batch_pred_probC = batch_pred_prob[1]

            # Calculate the loss value.
            if config.test:
                batch_label = np.array(batch_label)
                batch_label = batch_label.transpose((1, 0, 2, 3))
                cur_step_loss = np.sum((batch_label - batch_pred_prob)*(batch_label - 
                                        batch_pred_prob), axis=2)
                cur_step_loss = np.mean(cur_step_loss, axis=0)
                cur_step_loss = np.mean(cur_step_loss, axis=1)
                sum_test_loss += np.sum(cur_step_loss)

            # Save the results.
            print('\n', 'Saving results ...') 
            for i in range(len(batch_list[k])):
                file = batch_list[k][i]
                print(i + k*config.batch_size + 1, file)
                sta1 = file.split('.')[0]
                sta2 = file.split('.')[1]
                loc1 = station[sta1]
                loc2 = station[sta2]
                dist = (111*((loc1[0] - loc2[0])**2 + ((loc1[1] - loc2[1])*math.cos(loc1[0]*
                        3.14/180))**2)**0.5)
                if auto_refT:                
                    config.ref_T = min(config.range_T[2]-1, round((dist/1.5/3.2 - config.range_T[0])/config.dT))

                # Extract group velocity.
                dir_name = config.result_path + '/test_result/' + file
                os.system('mkdir -p %s' % dir_name)
                fig_name = '{}/test_result/{}/{}'.format(config.result_path, file, file)

                # np.savetxt('{}/mapG.{}.txt'.format(dir_name, file), batch_pred_probG[i])
                if config.test:
                    true_vG = batch_true_vG[i]
                pred_probG = np.array(batch_pred_probG[i])
                max_probG = np.max(pred_probG, axis=0)
                disp_G = batch_input[i, :, :, 0]
                pred_vG = []
                for j in range(len(max_probG)):
                    if max_probG[j] > config.confidence_G:
                        column = pred_probG[:, j]
                        # Find the index of the max value. Average from top to bottom 
                        # and from bottom to top.  
                        index1 = list(column).index(max_probG[j])
                        index2 = config.input_size[0] - 1 - list(column[::-1]).index(max_probG[j])
                        if disp_G[int((index1 + index2)/2), j] > config.disp_G_value:
                            pred_vG.append((index1 + index2)/2*config.dV)
                        else:
                            pred_vG.append(0)
                    else:
                        pred_vG.append(0)

                x = np.linspace(config.range_T[0], config.range_T[1], config.range_T[2])

                pred_vG = self.process_G(pred_vG, disp_G, pred_probG)

                # Find the T start and the end of the disp.
                # G_start is the T start of the group velocity.
                # G_end is the T end of the group velocity.
                G_start = config.input_size[1] - 1
                G_end = 0
                for j in range(len(pred_vG)):
                    if pred_vG[j] != 0:
                        G_start = min(G_start, j)
                        G_end = max(G_end, j)
                        pred_vG[j] += config.range_V[0]
                pred_vG = self.dist_constraint(pred_vG, dist)
                
                output_vG = []
                output_vG.append(x)
                output_vG.append(pred_vG[:config.input_size[1]])
                output_vG = np.array(output_vG).T
                np.savetxt('{}/newG.{}.txt'.format(dir_name, file), output_vG, fmt="%1.2f  %1.2f")

                # Extract phase velocity.
                # np.savetxt('{}/mapC.{}.txt'.format(dir_name, file), batch_pred_probC[i])
                pred_probC = np.array(batch_pred_probC[i])
                disp_C = batch_input[i, :, :, 1]
                if config.test:
                    true_vC = batch_true_vC[i]

                # random plot
                rand = list(np.zeros(int(1/config.random_plot) - 1))
                rand.append(1.0)
                plot = False
                key = random.sample(rand, 1)
                if key[0] == 1:
                    plot = True

                if config.ref_T2:
                    pred_vC = self.pick_C(pred_probC, config.ref_T2[0], config.ref_T2[1], disp_C, 
                                              fig_name, plot=plot)
                else:
                    pred_vC = self.pick_C(pred_probC, G_start, G_end, disp_C, fig_name, plot=plot)

                pred_vC = self.process_C(pred_vC, pred_probC)
                for j in range(len(pred_vC)):
                    if pred_vC[j] != 0:
                        pred_vC[j] += config.range_V[0]
                pred_vC = self.dist_constraint(pred_vC, dist)
                
                output_vC = []
                output_vC.append(x)
                output_vC.append(pred_vC[:config.input_size[1]])
                output_vC = np.array(output_vC).T
                np.savetxt('{}/newC.{}.txt'.format(dir_name, file), output_vC, fmt="%1.2f  %1.2f")

                # random plot
                if plot:
                    print('Plot', dir_name)
                    if config.test:
                        plot_test(disp_G, pred_probG, pred_vG, disp_C, pred_probC, pred_vC,
                                      fig_name, config.test, true_vG, true_vC)
                    else:
                        plot_test(disp_G, pred_probG, pred_vG, disp_C, pred_probC, pred_vC,
                                      fig_name, config.test)
                print('\n')

        if config.test:
            test_loss = sum_test_loss/file_num
            print('Test loss:', test_loss, '\n')


    def process_G(self, pred_vG, disp_G, pred_probG):
        """ Process the exrtacted group velocity dispersion curves.

        Args:
            pred_vG: Group velocity curve.
            disp_G: Group velocity dispersion image.
            pred_probG: Predicted group velocity probability map.
        
        Returns:
            Processed group velocity dispersion curve.

        """
        config = self.config
        max_dv = config.max_dv_G
        v_max = config.v_max
        v_min = config.v_min
        row = config.input_size[0]
        col = config.input_size[1]
        slow = config.slow_G
        dV = config.dV
        begin = config.begin                  # Use b to end to trace 0-b
        forward = config.forward
        backward = config.backward

        # index1: From short period to long period.
        # index2: From long period to short period.
        index1 = np.arange(0, col, 1)
        index2 = np.arange(col - 1, -1, -1)
        start = 0
        end = col - 1

        # Process1: remove outliers.
        for j in index1[1:col - 1]:
            if abs(pred_vG[j] - pred_vG[j - 1]) > max_dv and abs(pred_vG[j] - pred_vG[j + 1]) > max_dv:
                pred_vG[j] = 0.5*(pred_vG[j - 1] + pred_vG[j + 1])

        # Correction
        for j in index1:
            j = int(j)
            max_probGrange = int((j/col*1 + 0.1)/dV)     # Search range.
            if pred_vG[j] + config.range_V[0] > v_min and pred_vG[j] + config.range_V[0] < v_max:
                key_index = int(round(pred_vG[j]/dV))
                for k in range(max_probGrange):
                    if key_index - k > 0 and key_index + k < config.range_V[2] - 1:
                        if (disp_G[key_index + k, j] >= disp_G[key_index + k - 1, j] and 
                                disp_G[key_index + k, j] >= disp_G[key_index + k + 1, j]):
                            key_index = key_index + k
                            break
                        if (disp_G[key_index - k, j] >= disp_G[key_index - k - 1, j] and 
                                disp_G[key_index - k, j] >= disp_G[key_index - k + 1, j]):
                            key_index = key_index - k
                            break
                pred_vG[j] = key_index*dV

        # process2: remove unstable and only keep stable part.
        good_points = 0
        for j in index1[:col - 1]:
            if (abs(pred_vG[j + 1] - pred_vG[j]) > max_dv or pred_vG[j + 1] + config.range_V[0] > v_max or 
                    pred_vG[j + 1] + config.range_V[0] < v_min):
                start = j + 1
                good_points = 0
            else:
                good_points = good_points + 1
                if good_points >= 10:
                    break

        good_points = 0
        for j in index2[:col - 1]:
            if (abs(pred_vG[j] - pred_vG[j - 1]) > max_dv or pred_vG[j] + config.range_V[0] > v_max or 
                    pred_vG[j] + config.range_V[0] < v_min):
                end = j - 1
                good_points = 0
            else:
                good_points = good_points + 1
                if good_points >= 10:
                    break
        print('Group velocity period index range:', start, end)

        if start > 0:
            pred_vG[:start] = np.zeros(start)
        if end < col - 1:
            pred_vG[end + 1:] = np.zeros(col - 1 - end)

        # Process3: find the most stable stage.
        stage_index = []
        pred_vG = list(pred_vG)
        pred_vG.append(0)
        for j in index1:
            if pred_vG[j] != 0:
                if len(stage_index) == 0:
                    stage_index.append(j)
                if abs(pred_vG[j] - pred_vG[j + 1]) > max_dv:
                    stage_index.append(j + 1)
        if len(stage_index) == 1:
            stage_index.append(col)
        if len(stage_index) == 0:
            stage_index.append(0)
            stage_index.append(col)

        stage_length = list(np.array(stage_index[1:]) - np.array(stage_index[:-1]))
        len_stage = len(stage_index)

        # Calculate the average probability for each stage
        stage_eng = []
        for j in range(len(stage_index) - 1):
            eng = 0
            length = 0
            for k in range(stage_index[j], stage_index[j + 1]):
                eng += pred_probG[int(pred_vG[k]/dV), k]
                length += 1
            if length >= config.min_len:
                stage_eng.append(eng/length)
            else:
                stage_eng.append(0)
        max_probGeng = np.max(stage_eng)
        max_probGindex = stage_eng.index(max_probGeng)
        end = stage_index[max_probGindex + 1]
        start = stage_index[max_probGindex]
        new_pred_vG = np.zeros(col)

        # no smooth
        if len_stage <= 20 and np.max(stage_length) >= config.min_len:
            print('Average value in G dispersion image:', max_probGeng)
            if max_probGeng >= config.mean_confidence_G:
                new_pred_vG[max(start, begin):end] = pred_vG[max(start, begin):end]

                # Extend group velocity based on the stable stage.
                if max(start, begin) > 0 and forward:
                    i = list(range(max(start, begin)))
                    i.reverse()
                    for j in i:
                        key_index = int(round(new_pred_vG[j + 1]/dV))
                        max_probGrange = int(0.10/dV)
                        for k in range(max_probGrange):
                            if key_index - k > 0 and key_index + k < config.range_V[2] - 1:
                                if (disp_G[key_index - k, j] >= disp_G[key_index - k - 1, j] and 
                                        disp_G[key_index - k, j] >= disp_G[key_index - k + 1, j]):
                                    key_index = key_index - k
                                    break
                                if k >= slow:
                                    if disp_G[key_index + k - slow, j] >= disp_G[key_index + k - slow - 1, j] and \
                                            disp_G[key_index + k - slow, j] >= disp_G[key_index + k - slow + 1, j]:
                                        key_index = key_index + k - slow
                                        break

                        if disp_G[key_index,j] >= config.disp_G_value and k < max_probGrange - 1:
                            new_pred_vG[j] = key_index*dV
                        else:
                            break

                if end < col - 1 and backward:
                    i = list(range(col))[end:]
                    for j in i:
                        key_index = int(round(new_pred_vG[j - 1]/dV))
                        # max_probGrange = int((j/col*1.0 + 0.1)*500)
                        max_probGrange = int(0.12/dV)    # Search in 0.1 range
                        for k in range(max_probGrange):
                            if key_index - k > 0 and key_index + k < config.range_V[2] - 1:
                                if (disp_G[key_index + k, j] >= disp_G[key_index + k - 1, j] and 
                                        disp_G[key_index + k, j] >= disp_G[key_index + k + 1, j]):
                                    key_index = key_index + k
                                    break
                                if k >= slow:
                                    if (disp_G[key_index - k + slow, j] >= disp_G[key_index - k + slow - 1, j] and
                                            disp_G[key_index - k + slow, j] >= disp_G[key_index - k + slow + 1, j]):
                                        key_index = key_index - k + slow
                                        break

                        if disp_G[key_index, j] >= config.disp_G_value and k < max_probGrange - 1:
                            new_pred_vG[j] = key_index*dV
                        else:
                            break
        return new_pred_vG


    def pick_C(self, map_C, start_T, end_T, disp_C, name, plot):
        """Extracted phase velocity dispersion curves.

        Args:
            map_C: Phase velocity probability image.
            start_T，end_T: Use these columns to calculate the average probability of C curves.
            disp_C: Phase velocity disprsion image.
            name: Predicted group velocity probability map.
            plot: Whether to plot.
        
        Returns:
            Extracted raw phase velocity dispersion curve.
        """
        config = self.config
        col = config.input_size[1]
        name = name + '_pc.jpg'
        dV = config.dV
        dT = config.dT
        up = 6                   # up the lower boundary

        # find potential phase curve
        ref_points = []   # size: n*1
        ref_col = disp_C[:, config.ref_T]
        slow = config.slow_C
        for j in range(1, config.range_V[2] - 1):
            if ref_col[j] >= ref_col[j - 1] and ref_col[j] >= ref_col[j + 1]:
                if ref_col[j] == ref_col[j - 1]:
                    ref_points.append(j - 1)
                else:
                    ref_points.append(j)

        potential_c = []
        # print(ref_points)

        for each_refp in ref_points:

            each_curve = [[config.ref_T, each_refp]]
            # trace before the reference point
            before = list(range(config.ref_T))
            before.reverse()
            key_index = each_refp
            for j in before:        # loop for each column
                max_probGrange = int((j/col*1.0 + 0.3)/dV)
                if key_index > config.range_V[2] - 1 or key_index < 0:
                    break
                if (disp_C[key_index, j] >= disp_C[key_index + 1, j] and disp_C[key_index, j] >= 
                        disp_C[key_index - 1, j]):
                    if disp_C[key_index, j] == disp_C[key_index - 1, j]:
                        key_index = key_index - 1
                    else:
                        key_index = key_index
                    each_curve.insert(0, [j, key_index])
                else:
                    for k in range(max_probGrange)[1:]:
                        exist = False
                        if key_index - k > 0 and key_index - k < config.range_V[2] - 1:
                            if (disp_C[key_index - k, j] >= disp_C[key_index - k - 1, j] and 
                                    disp_C[key_index - k, j] >= disp_C[key_index - k + 1, j]):
                                key_index = key_index - k
                                each_curve.insert(0, [j, key_index])
                                exist = True
                                # print(fig[key_index - 1, j],fig[key_index, j],fig[key_index + 1, j])
                                # print(key_index, j)
                                break
                        if k >= slow:
                            if key_index + k - slow < config.range_V[2] - 1 and key_index + k - slow >= 0:
                                if (disp_C[key_index + k - slow, j] >= disp_C[key_index + k - slow - 1, j] and 
                                        disp_C[key_index + k - slow, j] >= disp_C[key_index + k - slow + 1, j]):
                                    key_index = key_index + k - slow
                                    each_curve.insert(0, [j, key_index])
                                    exist = True
                                    # print(fig[key_index - 1, j],fig[key_index, j],fig[key_index + 1, j])
                                    # print(key_index + k, j)
                                    break
                    if not exist:
                        break
                # boundary stop
                if key_index <=  up or key_index >= config.range_V[2] - 1:
                    break

            # Trace after the reference point
            after = list(range(col)[config.ref_T + 1:])
            key_index = each_refp
            for j in after:        # loop for each column
                max_probGrange = int((j/col*1.0 + 0.3)/dV)
                if disp_C[key_index, j] >= disp_C[key_index + 1, j] and disp_C[key_index, j] >= disp_C[key_index - 1, j]:
                    if disp_C[key_index, j] == disp_C[key_index + 1, j]:
                        key_index = key_index + 1
                    else:
                        key_index = key_index
                    each_curve.append([j, key_index])
                else:
                    for k in range(max_probGrange)[1:]:
                        exist = False
                        if key_index + k > 0 and key_index + k < config.range_V[2] - 1:
                            if disp_C[key_index + k, j] >= disp_C[key_index + k - 1, j] and \
                                    disp_C[key_index + k, j] >= disp_C[key_index + k + 1, j]:
                                key_index = key_index + k
                                each_curve.append([j, key_index])
                                exist = True
                                # print(fig[key_index - 1, j],fig[key_index, j],fig[key_index + 1, j])
                                # print(key_index, j)
                                break
                        if k >= slow:
                            if key_index - k + slow > 0 and key_index - k + slow < config.range_V[2] - 1:
                                if disp_C[key_index - k + slow, j] >= disp_C[key_index - k + slow - 1, j] and \
                                        disp_C[key_index - k + slow, j] >= disp_C[key_index - k + slow + 1, j]:
                                    key_index = key_index - k + slow
                                    each_curve.append([j, key_index])
                                    exist = True
                                    # print(fig[key_index - 1, j],fig[key_index, j],fig[key_index + 1, j])
                                    # print(key_index + k, j)
                                    break
                    if not exist:
                        break

                # boundary stop
                if key_index <= up + 1  or key_index >= config.range_V[2] - 2:
                    break
            potential_c.append(each_curve)

        # plot each potential phase curve
        if plot and len(potential_c) != 0:
            plt.figure(figsize=(5, 3), clear=True)
            plt.tight_layout()
            fontsize = 12
            figformat = '.png'

            x1 = np.linspace(config.range_T[0], config.range_T[1], config.range_T[2])
            y1 = np.linspace(config.range_V[0], config.range_V[1], config.range_V[2])
            plt.pcolor(x1, y1, disp_C, shading='auto', cmap='jet', vmin=-1, vmax=1.05)
            plt.colorbar()
            plt.xlabel('Period (s)',fontsize=fontsize)
            plt.ylabel('Phase Velocity (km/s)',fontsize=fontsize)
            plt.title('C spectrogram',fontsize=fontsize)

            for each in potential_c:
                # print(each)
                y2 = np.array(each)[:, 1]*dV + np.ones(len(each))*config.range_V[0]
                x2 = np.array(each)[:, 0]*dT + config.range_T[0]
                plt.plot(x2, y2, '-wo', markersize=1)

            for i in ref_points:
                plt.plot(x1[config.ref_T], i*dV + config.range_V[0], 'wo', markersize=5)

            # plt.show()
            plt.savefig(name, bbox_inches='tight', dpi=300)
            plt.close()
        print('Potential phase curve number:', len(potential_c))

        # find the best phase curve
        pred_vG = np.zeros(col)
        potential_c2 = []
        if len(potential_c) != 0:
            confidence = []
            each_len = []
            for each in potential_c:
                each_conf = 0
                num = 0
                for each_item in each:
                    if each_item[0] >= start_T and each_item[0] <= end_T:
                        each_conf += map_C[each_item[1], each_item[0]]
                        num += 1
                if num >= config.min_len:
                    confidence.append(each_conf)
                    potential_c2.append(each)
                    each_len.append(num)
            # print(len(potential_c2),len(potential_c))
            # print(confidence, each_len)
            if len(confidence) != 0:
                max_probGindex = confidence.index(max(confidence))
                print('Max average C probability value:', confidence[max_probGindex]/each_len[max_probGindex])
                if confidence[max_probGindex]/each_len[max_probGindex] >= config.mean_confidence_C:
                    for each2 in potential_c2[max_probGindex]:
                        pred_vG[each2[0]] = each2[1]*dV
        return pred_vG


    def process_C(self, pred_vC, map_C):
        """Process phase velocity dispersion curves.

        Args:
            pred_vC: Phase velocity dispersion curve.
            map_C: Phase velocity probability image.
        
        Returns:
            Processed phase velocity dispersion curve.
        """
        config = self.config
        max_dv = config.max_dv_C
        v_max = config.v_max
        v_min = config.v_min
        row = config.input_size[0]
        col = config.input_size[1]
        slow = config.slow_G
        dV = config.dV
        dT = config.dT

        # process
        index1 = np.arange(0, col, 1)
        index2 = np.arange(col - 1, -1, -1)
        start = 0
        end = col - 1

        # process1: remove unstable and only save stable part
        good_points = 0
        for j in index1[:col - 1]:
            if (abs(pred_vC[j + 1] - pred_vC[j]) > max_dv or pred_vC[j + 1] + config.range_V[0] > 
                    v_max or pred_vC[j + 1] + config.range_V[0] < v_min):
                start = j + 1
                good_points = 0
            else:
                good_points = good_points + 1
                if good_points >= 10:
                    break
        good_points = 0
        for j in index2[:col - 1]:
            if (abs(pred_vC[j] - pred_vC[j - 1]) > max_dv or pred_vC[j] + config.range_V[0] > 
                    v_max or pred_vC[j] + config.range_V[0] < v_min):
                end = j - 1
                good_points = 0
            else:
                good_points = good_points + 1
                if good_points >= 8:
                    break

        if start > 0:
            pred_vC[:start] = np.zeros(start)
        if end < col - 1:
            pred_vC[end + 1:] = np.zeros(col - 1 - end)

        # process2: remove unstable and only keep stable part.
        stage_index = []
        pred_vC = list(pred_vC)
        pred_vC.append(0)
        for j in index1[:col]:
            if pred_vC[j] != 0:
                if len(stage_index) == 0:
                    stage_index.append(j)
                if abs(pred_vC[j] - pred_vC[j + 1]) > max_dv:
                    stage_index.append(j + 1)
        if len(stage_index) == 1:
            stage_index.append(col)
        if len(stage_index) == 0:
            stage_index.append(0)
            stage_index.append(col)

        stage_length = list(np.array(stage_index[1:]) - np.array(stage_index[:-1]))
        len_stage = len(stage_index)
        stage_eng = []
        for j in range(len(stage_index) - 1):
            sum_eng = 0
            start = stage_index[j]
            end = stage_index[j + 1]
            num = stage_length[j]
            for k in range(start, end):
                #index = int(pred_vC[k]/dV)
                #print(index)
                sum_eng += map_C[int(pred_vC[k]/dV)][k]
            if num >= config.min_len:
                stage_eng.append(sum_eng/num)
            else:
                stage_eng.append(0)

        max_probGindex = stage_eng.index(np.max(stage_eng))
        start = stage_index[max_probGindex]
        end = stage_index[max_probGindex + 1]
        for k in range(end - 1, -1, -1):
            if map_C[int(pred_vC[k]/dV)][k] >= config.confidence_C:
                break
        new_pred_vG = np.zeros(col)
        print('Phase velocity period index range:', start, k)
        # print(stage_index, stage_eng)
        # no smooth
        if len_stage <= 5 and stage_length[max_probGindex] >= config.min_len:
            new_pred_vG[start:k + 1] = pred_vC[start:k + 1]

        # new_pred_vG = pred_vC
        return new_pred_vG


    def dist_constraint(self, pred_v: np.array, dist: float):
        """Add distance constraint to extracted dispersion curves.
        Distance must be larger than 1.5*v*T

        Args:
            pred_v: Extracted dispersion curve.
            dist: Station pair distance.
        
        Returns:
            Dispersion curves with distance constraint.
        """
        config = self.config
        range_T = config.range_T
        T = np.linspace(range_T[0], range_T[1], range_T[2])
        for i in range(len(T)):
            if pred_v[i] != 0:
                if dist/1.5/pred_v[i] < T[i]:
                    break
        new_curve = np.zeros(len(T))
        new_curve[:i + 1] = pred_v[:i + 1]

        return new_curve


if __name__ == '__main__':
    Pick_v().pick_v()
    # Summarize the dispersion curves and plots into corresponding folders.
    os.system('python %s/cp_png.py'%Config().result_path)
    os.system('python %s/cp_v.py'%Config().result_path)
