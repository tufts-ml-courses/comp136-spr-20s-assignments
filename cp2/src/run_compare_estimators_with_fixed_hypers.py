'''
Summary
-------
This script produces two figures for each dataset

1. Plot of outputs vs inputs
2. Plot of performance score vs. num training data seen

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from FeatureTransformPolynomial import PolynomialFeatureTransform
from LinearRegressionMAPEstimator import LinearRegressionMAPEstimator
from LinearRegressionPosteriorPredictiveEstimator import LinearRegressionPosteriorPredictiveEstimator

if __name__ == '__main__':
    
    order_list = [0, 1, 2, 3, 4]
        
    n_train_list = [0, 8, 64, 512]
        
    alpha = 1.0 # moderate prior precision
    beta = 20.0 # strong likelihood precision

    for dataset_name in ['toyline', 'toywave']:
        train_df = pd.read_csv("../data/%s_train.csv" % dataset_name)
        test_df = pd.read_csv("../data/%s_test.csv" % dataset_name)

        x_train_ND, t_train_N = train_df['x'].values[:,np.newaxis], train_df['y'].values
        x_test_ND, t_test_N = test_df['x'].values[:,np.newaxis], test_df['y'].values

        fig1, perf_vs_N__ax_grid = plt.subplots(nrows=1, ncols=4,
            sharex=True, sharey=True, squeeze=True,
            figsize=(4 * len(order_list), 4))
        
        fig2, y_vs_x__ax_grid = plt.subplots(nrows=1, ncols=4,
            sharex=True, sharey=True, squeeze=True,
            figsize=(4 * len(order_list), 4))
        
        for order, perfvsN_ax, xy_ax in zip(order_list, perf_vs_N__ax_grid, y_vs_x__ax_grid):
            feature_transformer = PolynomialFeatureTransform(order=order, input_dim=1)

            map_train_scores = np.zeros(len(n_train_list))
            map_test_scores = np.zeros(len(n_train_list))
            ppe_train_scores = np.zeros(len(n_train_list))
            ppe_test_scores = np.zeros(len(n_train_list))

            print("===== MAPEstimator with alpha %.3g, beta %.3g" % (alpha, beta))
            for ff, N in enumerate(n_train_list):
                estimator = LinearRegressionMAPEstimator(feature_transformer, alpha=alpha, beta=beta)
                ## TODO fit estimator on first N examples in train
                ## TODO record estimator's score on train in map_train_scores
                ## TODO record estimator's score on test in map_test_scores
                print("%6d examples : train score % 9.3f | test score % 9.3f" % (
                    N, map_train_scores[ff], map_test_scores[ff]))

            print("===== PosteriorPredictiveEstimator with alpha %.3g, beta %.3g" % (alpha, beta))
            for ff, N in enumerate(n_train_list):
                ppe_estimator = LinearRegressionPosteriorPredictiveEstimator(feature_transformer, alpha=alpha, beta=beta)
                ## TODO fit estimator on first N examples in train
                ## TODO record estimator's score on train
                ## TODO record estimator's score on test
                print("%6d examples : train score % 9.3f | test score % 9.3f" % (
                    N, ppe_train_scores[ff], ppe_test_scores[ff]))

            # Plot on log scale (manually crafted)
            int_list = np.arange(len(n_train_list))
            perfvsN_ax.plot(int_list, map_test_scores, 'b.-', label='MAP estimator')
            perfvsN_ax.plot(int_list, ppe_test_scores, 'g.-', label='PosteriorPredictive estimator')
            perfvsN_ax.legend(loc='lower right')
            # Manually crafted x scale
            perfvsN_ax.set_xticks([a for a in int_list])
            perfvsN_ax.set_xticklabels(['%d' % a for a in n_train_list])

            ## Plot inputs vs predictions
            xy_ax.plot(x_train_ND[:,0], t_train_N, 'k.', alpha=0.3)
            G = 200 # num grid points
            xmin = x_train_ND[:,0].min()
            xmax = x_train_ND[:,0].max()
            R = xmax - xmin
            xgrid_G = np.linspace(xmin - R, xmax + R, G)
            xgrid_G1 = np.reshape(xgrid_G, (G, 1))

            ## TODO compute mean prediction at each entry of the grid
            mean_G = np.zeros(G)
            ## TODO compute stddev of prediction at each entry of grid
            stddev_G = np.ones(G)

            ## Plot the mean as solid line, plus light fill for (-3, +3 stddev) range
            xy_ax.fill_between(xgrid_G, mean_G -3 * stddev_G, mean_G +3 * stddev_G,
                facecolor='blue', alpha=0.2, label='3 stddev range')
            xy_ax.plot(xgrid_G, mean_G, 'b.-', label='prediction')
            xy_ax.legend(loc='lower right')
            xy_ax.set_ylim([-5, 5])
            
    plt.show()
