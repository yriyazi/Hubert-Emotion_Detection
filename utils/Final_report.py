# import numpy as np
# import pandas                   as pd
# import torch
# import dataloaders
# import utils

# def report(Cifar10CnnModel:torch.nn.Module,
#            Model_name:str,
#            read_df_file:any=None):
#     model1 = Cifar10CnnModel()
#     model1.load_state_dict(torch.load("./Model/"+Model_name+"/"+Model_name+".pt"))
#     model1.eval()

#     utils.plot.prediction(model=model1,classes=utils.classes,model_name=Model_name+"_test ",teste_data_sets=dataloaders.teste_data_sets)
#     utils.plot.prediction(model=model1,classes=utils.classes,model_name=Model_name+"_train ",teste_data_sets=dataloaders.train_data_sets)

#     if read_df_file==None:
#         df = pd.read_csv('./Model/'+Model_name+'/'+Model_name+'_report.csv')
#     else:
#         df = read_df_file
#     # epochs=160

#     train=df.query('mode == "train"').query('batch_index == 23')
#     validation=df.query('mode == "val"').query('batch_index == 25')
#     test=df.query('mode == "test"').query('batch_index == 20')

#     utils.plot.result_plot2(Model_name+"_Accuracy","Accuracy",
#                             np.array(train['avg_train_acc_till_current_batch']),
#                             np.array(validation['avg_val_acc_till_current_batch']),
#                             np.array(test['avg_val_acc_till_current_batch']),
#                             DPI=400)

#     utils.plot.result_plot2(Model_name+"_loss","loss",
#                             np.array(train['avg_train_loss_till_current_batch']),
#                             np.array(validation['avg_val_loss_till_current_batch']),
#                             np.array(test['avg_val_loss_till_current_batch']),
#                             DPI=400)