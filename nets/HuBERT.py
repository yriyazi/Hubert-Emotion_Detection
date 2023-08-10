import utils
from transformers import AutoConfig, AutoTokenizer, AutoModel

model_name  = "facebook/hubert-base-ls960"
HuBERT      = AutoModel.from_pretrained(model_name,output_hidden_states= utils.output_hidden_states)