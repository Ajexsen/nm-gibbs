#!/usr/bin/python
import sys
import datetime, os

import numpy as np
import pandas as pd
import torch

from Yuro_function import *

import logging
import time


def infer(nets, ibp, training_iterations):
    args_dataset = 'mnist'
    args_datadir = 'data/mnist/'
    train_dl, test_dl = get_dataloader(args_dataset, args_datadir, 32, 32)
    logging.getLogger("pulp").setLevel(logging.INFO)

    k_track = []
    log_track = []
    sigma_x = []
    accuracy = []

    for iteration in range(training_iterations):
        start = time.time()
        log_likelihood = ibp.learning
        n_classes = 10
        end = time.time()
        logging.info("------- iteration: {}\tK: {}\tlikelihood: {} \tDuration: {}".format(ibp._counter, ibp._K, log_likelihood,
                                                                                   (end - start)))
        print("sigma_a:", ibp._sigma_a)  # , ", sigma_x:", ibp._sigma_x)
        print("sigma_xj:", ibp._sigma_xj)
        print("Calculate current global accuracy.")
        start = time.time()
        train_acc, test_acc = compute_net_accuracy(nets, ibp.get_A(), ibp.get_A_sigma(), n_classes, train_dl, test_dl)
        end = time.time()
        print("train acc: {}, test_acc: {}, compute accuracy duration: {}".format(train_acc, test_acc, (end - start)))
        start = time.time()
        train_acc_u, test_acc_u = compute_net_accuracy_uniform(nets, ibp.get_A(), ibp.get_A_sigma(), n_classes, train_dl, test_dl)
        end = time.time()
        print("(UNIFORM) train acc: {}, test_acc: {}, compute accuracy duration: {}".format(train_acc_u, test_acc_u, (end - start)))
        print("-------------------------------------")

        k_track.append(ibp._K)
        log_track.append(log_likelihood)
        sigma_x.append(ibp._sigma_xj)
        accuracy.append((train_acc, test_acc))


def main():
    logging.basicConfig(level=logging.DEBUG)  # Set this to INFO level, when doing larger experiments. Logging DEBUG information to console slows down the execution.
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    mpl_log = logging.getLogger('matplotlib')
    mpl_log.setLevel(logging.WARNING)
    np.set_printoptions(precision=12)

    isInitialized = False
    # LINE SEARCH: N, NP, H, N (sigma_x), BETA
    for s_i in range(1, 2, 1):
        logging.info("Experiment #{}".format(s_i))

        if isInitialized is False:
            try:
                beta_hyper_parameter = (.001, .001)
                sigma_x_hyper_parameter = (1., 1.)

                # <SIMULATE DATA>
                layer_i = 0
                n_model = 5
                n_neuron = 15
                print("Loading nets")
                # current_path = os.path.dirname(__file__)
                model_path = 'model/' + str(n_model) + '_' + str(n_neuron)

                nets = [None for i in range(n_model)]

                for net_id in range(n_model):
                    nets[net_id] = torch.load(model_path + "/model-" + str(net_id))
                    # statedict = nets[net_id].state_dict()
                    # layer_weight = statedict['layers.%d.weight' % layer_i].numpy().T
                    # layer_bias = statedict['layers.%d.bias' % layer_i].numpy()
                    # # net_weights.extend([layer_weight, layer_bias])
                    # # weights.append(net_weights)
                    # weights.append(np.vstack((layer_weight, layer_bias)).T)
                batch_weights = pdm_prepare_weights(nets)
                print("Nets loaded")

                J = len(batch_weights)
                L_next = None
                c = 1
                assignment_c = [None for j in range(J)]
                weights_bias = [np.hstack((batch_weights[j][0].T, batch_weights[j][c * 2 - 1].reshape(-1, 1),
                                           # .reshape(-1, 1) -> transpose
                                           patch_weights(batch_weights[j][c * 2], L_next, assignment_c[j]))) for j in
                                range(J)]

                train_data = weights_bias
                # </SIMULATE DATA>
                isInitialized = True

            except Exception as e:
                logging.debug("Init failed.")
                pass

            # <Inference>
            # import uncollapsed_gibb_sampleB_FULL as uncollapsed_gibbs
            import uncollapsed_gibb_sampleB as uncollapsed_gibbs
            # import uncollapsed_gibb_mod as uncollapsed_gibbs

            training_iterations = 15
            alpha_alpha = 15
            beta_beta = 5
            sigma_a = 1
            sigma_x = 1
            # alpha_alpha = 20
            # beta_beta = 4

            ibp_inferencer = uncollapsed_gibbs.UncollapsedGibbs(beta_hyper_parameter=beta_hyper_parameter,
                                                                sigma_x_hyper_parameter=sigma_x_hyper_parameter)
            ibp_inferencer._initialize(train_data, alpha_alpha, beta_beta, sigma_a, sigma_x)
            start_all = time.time()
            infer(nets, ibp_inferencer, training_iterations)
            end_all = time.time()
            result = ibp_inferencer.result()
            print("end in: {}".format((end_all - start_all)))
            print()


if __name__ == '__main__':
    main()
