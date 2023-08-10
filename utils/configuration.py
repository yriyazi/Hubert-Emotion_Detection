import yaml

# Load config file
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Access hyperparameters
inference_mode      = config['inference_mode']
learning_rate       = config['learning_rate']
num_epochs          = config['num_epochs']
seed                = config['seed']
ckpt_save_freq      = config['ckpt_save_freq']

# Access dataset parameters
dataset_path        = config['dataset']['path']
train_split         = config['dataset']['train_split']
validation_split    = config['dataset']['validation_split']
test_split          = config['dataset']['test_split']
classes             = config['dataset']['classes']



Tokenizer_Max_lenght= config['Tokenizer']['Max_lenght']


# Access model architecture parameters
model_name                  = config['model']['name']
pretrained                  = config['model']['pretrained']
num_classes                 = config['model']['num_classes']
Wave2vec                    = config['model']['Wave2vec']
output_hidden_states        = config['model']['output_hidden_states']
Act                         = config['model']['Act']
hidden_states_layer_number  = config['model']['hidden_states_layer_number']

# Access optimizer parameters
optimizer_name      = config['optimizer']['name']
# weight_decay        = config['optimizer']['weight_decay']
# opt_momentum        = config['optimizer']['momentum']
# Access scheduler parameters
scheduler_name  = config['scheduler']['name']
start_factor    = config['scheduler']['start_factor']
end_factor      = config['scheduler']['end_factor']


# print("configuration hass been loaded!!! \n successfully")
# print(learning_rate)