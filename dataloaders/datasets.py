import  torch
import  torchaudio
from    torch.utils.data        import Dataset, random_split
from    sklearn.model_selection import train_test_split
from    .preprocess             import preprocess_audio
from torchvision import transforms

# Define your dataset class
class CustomDataset(Dataset):
    def __init__(self, 
                 data, 
                 targets,
                 Wave2Vec:bool = False):
        
        self.data       = data
        self.targets    = targets
        self.Wave2Vec   = Wave2Vec
        
        self.new_sample_rate = 16000
        
        desired_duration = 4 #was 2 seconds
        desired_samples = int(desired_duration * self.new_sample_rate)
        self.transform = transforms.Compose([transforms.Lambda(lambda x: x[:, :desired_samples] if x.size(1) >= desired_samples else x)])

        
    def __getitem__(self, index):
        if self.Wave2Vec==True:
            x, sample_rate = torchaudio.load(self.data[index])
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                                       new_freq=self.new_sample_rate)
            x = resampler(x)            
            x = self.transform(x)

            y = self.targets[index]
            return x.squeeze(0), y
        
        x = preprocess_audio(self.data[index])
        y = self.targets[index]
        return x, y
    
    def __len__(self):
        return len(self.data)
    
def collate_fn(batch):
    """
    This is a function for collating a batch of variable-length sequences into a single tensor, which is
    useful for training a neural network with PyTorch.

    The input to this function is a batch of samples, each containing a source and target sequence. 
    The function extracts the source and target sequences from each sample, and then pads them to ensure
    that all sequences in the batch have the same length. This is necessary because PyTorch requires all
    inputs to a neural network to have the same shape.

    The function uses the PyTorch pad_sequence function to pad the sequences. pad_sequence is called with
    the batch_first=True argument to ensure that the batch dimension is the first dimension of the output
    tensor. The padding_value argument is set to 0 to pad with zeros.

    The function returns the padded source and target sequences as a tuple.
    """
    sources = [item[0] for item in batch]
    targets = [item[1] for item in batch]
              
    sources = torch.nn.utils.rnn.pad_sequence(sources, batch_first=True)
    # targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
    return sources, torch.tensor(targets)