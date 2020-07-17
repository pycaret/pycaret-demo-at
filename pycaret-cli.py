#getting input parameters from sys
import sys
dataset = sys.argv[1]
target = sys.argv[2]
exp_name = str(dataset) + '_exp'

#import dataset using sys arg
from pycaret.datasets import get_data
data = get_data(dataset)

#initialize setup
from pycaret.classification import setup, compare_models, blend_models, tune_model, save_model, deploy_model, automl
clf1 = setup(data, target = target, silent=True, html=False, log_experiment=True, experiment_name=exp_name)

#compare models and select top5
top5 = compare_models(n_select = 5, blacklist = ['catboost'])

#blend top 5 models
blender = blend_models(estimator_list = top5)

#tune best model
tuned_best_model = tune_model(top5[0])

#select best model
a = automl()
save_model(a, 'best_model')

#deploy best model
deploy_model(a, model_name='best-model-aws', authentication = {'bucket' : 'pycaret-test'})