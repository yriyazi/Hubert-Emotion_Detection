import glob
import os

def crawler(DIR_TRAIN='dataset/archive/',
            ):
    classes = os.listdir(DIR_TRAIN)
    
    Dump  = []
    Dump_class = []
    for _class in classes:
        Dump += glob.glob(DIR_TRAIN  + _class + '/*.wav')
    
    dictionary = {'N':0,
                  'A':1,
                  'W':2,
                  'H':3,
                  'F':4,
                  'S':5}
    
    for index in range(len(Dump)):
         Dump_class  .append(dictionary[Dump[index][-7:-6]])
       
    print("\nTotal : ", len(Dump))
 
    
    return Dump,Dump_class,dictionary