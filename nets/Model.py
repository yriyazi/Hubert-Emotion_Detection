import  torch,utils
import  torch.nn    as      nn
from    .HuBERT     import  HuBERT
from    transformers import Wav2Vec2FeatureExtractor


    
class Wave2VecClassifier(torch.nn.Module):
    def __init__(self,
                 HuBERT = HuBERT,
                 device:str = 'cuda'):
        
        super().__init__()
        self.device = device
        self.embedding  = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h')
        self.BERT       = HuBERT.to(device)
        
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(768,128),
                                        nn.ReLU(),
                                        nn.Linear(128,6))
        
        self.Act                        = utils.Act
        self.hidden_states_layer_numer  = utils.hidden_states_layer_number
        
    def Head(self,input):
        
        if    self.Act=='last_hidden_state':
            return input.last_hidden_state.mean(dim=1)        

        elif    self.Act=='hidden_states':
            Concat = []
            for index in (range(13-self.hidden_states_layer_numer,13,1)):
                Concat.append(input.hidden_states[index].mean(dim=-2))
            return torch.stack(Concat, dim=0, out=None).mean(dim=0)
        
    def forward(self, input):
        outputs = self.embedding(input,sampling_rate=16000, return_tensors='pt').input_values.to(self.device)        
        outputs = self.BERT(outputs.squeeze(0))
        outputs = self.Head(outputs)
        outputs = self.classifier(outputs)
        return outputs