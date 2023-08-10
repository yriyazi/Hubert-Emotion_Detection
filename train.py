import utils
utils.set_seed(utils.seed)
utils.Wave2vec = True

import  torch,tqdm,time,random
import  numpy               as      np
from    torch.utils.data    import  DataLoader

import dataloaders,nets,deeplearning
#%%
train_dataloader = DataLoader(dataloaders.train_dataset ,batch_size=32,shuffle=True,num_workers=2,collate_fn = dataloaders.collate_fn)
valid_dataloader = DataLoader(dataloaders.val_dataset   ,batch_size=64,shuffle=True,num_workers=2,collate_fn = dataloaders.collate_fn)
test_dataloader = DataLoader(dataloaders.test_dataset   ,batch_size=64,shuffle=True,num_workers=2,collate_fn = dataloaders.collate_fn)

#%%
model = nets.Wave2VecClassifier().to(utils.device)   

for param in model.BERT.parameters():
    param.requires_grad = False 
#%%
epoch           = utils.num_epochs
maxed_lr_bert = 5e-3

optimizer = torch.optim.AdamW(model.parameters(),
                            #   weight_decay = 8e-2
                              lr=  maxed_lr_bert)  
total_steps = len(train_dataloader) * epoch
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=maxed_lr_bert, 
                                                steps_per_epoch=len(train_dataloader),
                                                epochs=epoch)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.3)

ckpt_save_freq  = utils.ckpt_save_freq

Model_name = 'hubert-base-ls960'
model, optimizer, report = deeplearning.train(
    train_loader    = train_dataloader,
    val_loader      = valid_dataloader,
    model           = model,
    model_name      = Model_name + "_00_30",
    epochs          = epoch,
    
    load_saved_model    = False,
    ckpt_save_freq      = ckpt_save_freq,
    ckpt_save_path      = "Model/hubert-base-ls960/CheckPoints",
    ckpt_path           = "Model/hubert-base-ls960/CheckPoints/ckpt_hubert-base-ls960_30_60_epoch30.ckpt",
    report_path         = "Model/hubert-base-ls960",
    
    optimizer = optimizer,
    lr_schedulerr = scheduler,
    sleep_time = 10,
    validation_threshold = 77,
    
    test_ealuate    = True,
    tets_loader     = test_dataloader,
)