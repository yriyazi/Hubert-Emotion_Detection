import utils
utils.set_seed(utils.seed)
utils.Wave2vec = True

import nets , torch
import pandas as pd
import  dataloaders
import  numpy               as np
from    torch.utils.data    import  DataLoader
#%%
Model_name = 'hubert-base-ls960'
model_path = 'Model/'+Model_name+'/hubert-base-ls960_00_30_valid_acc 78.8352279663086.pt'
report_path = 'Model/'+Model_name+'/hubert-base-ls960_00_30_report.csv'

#%%
model = nets.Wave2VecClassifier().to(utils.device)
model .load_state_dict(torch.load(model_path))
# Set the model to evaluation mode
model.eval()

#%%
test__dataloader = DataLoader(dataloaders.test_dataset  ,batch_size=64,shuffle=True,num_workers=2,collate_fn = dataloaders.collate_fn)

utils.generate_classification_report(model=model,
                                     model_name = Model_name,
                                     dataloader = test__dataloader,
                                     class_names= ['Natural','Anger','Worried','Happy','Fear','Sadness'])
#%%

df = pd.read_csv(report_path)


def accu_2_int(df:pd.DataFrame,
               column:str='avg_train_acc_nopad_till_current_batch'):
    Dump = []
    for index in range(len(df)):
        Dump.append(float(np.array(df[column])[index][7:-18]))
    return Dump

train=df.query('mode == "train"').query('batch_index == 75')
test=df.query('mode == "val"').query('batch_index == 5')

utils.plot.result_plot(Model_name+"_Accuracy",
                       "Accuracy",
                        accu_2_int(train,'avg_train_acc_till_current_batch'),
                        accu_2_int(test,'avg_val_acc_till_current_batch'),
                        DPI=400)


utils.plot.result_plot(Model_name+"_loss",
                       "loss",
                        np.array(train['avg_train_loss_till_current_batch']),
                        np.array(test['avg_val_loss_till_current_batch']),
                        DPI=400)

#%%
