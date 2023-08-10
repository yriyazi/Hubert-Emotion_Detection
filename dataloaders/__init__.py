from .preprocess import *
from .datasets import CustomDataset,collate_fn
from .crawler  import crawler

#%%---------------------------------------------------------
import utils
from    collections             import Counter
from    sklearn.model_selection import train_test_split
from    torch.utils.data        import Dataset, random_split
#%%
Dump,Dump_class,dictionary = crawler()
# Split the data into train and test sets while stratifying
train_data, test_data, train_targets, test_targets = train_test_split(Dump, 
                                                                      Dump_class, 
                                                                      test_size=0.2, 
                                                                      stratify=Dump_class)

# Create dataset objects for train and test sets
train_dataset   = CustomDataset(train_data  , train_targets ,Wave2Vec=utils.Wave2vec)
test_dataset    = CustomDataset(test_data   , test_targets  ,Wave2Vec=utils.Wave2vec)


# Get the number of samples for each dataset
train_size  = len(train_dataset)
test_size   = len(test_dataset)

# Split the train dataset into train and validation sets while stratifying

test_dataset, val_dataset = random_split(test_dataset, [int(test_size/2), test_size-int(test_size/2)],
                                          generator=torch.Generator().manual_seed(42))


# # Print the number of samples in each set
# print(f"Train set size: {train_size}")
# print(f"Validation set size: {val_size}")
# print(f"Test set size: {test_size}")

#%% checking the count per class
# Count the occurrences of each element in the list
count_dict = Counter(train_targets)
print('\n')
for element, count in count_dict.items():
    print(f"train : {element}: {count}")
count_dict = Counter(test_targets)
print('\n')
for element, count in count_dict.items():
    print(f"test : {element}: {count}")


#%%
