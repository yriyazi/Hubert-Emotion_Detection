import  torch
import  numpy                   as np
import  matplotlib.pyplot       as plt
from    sklearn.metrics         import classification_report,confusion_matrix

def generate_classification_report(model,
                                   model_name,
                                   dataloader, 
                                   class_names):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for Data in dataloader:
            input_ids   = Data[0].to(device)
            labels      = Data[1].to(device)
            outputs     = model(input_ids)
            predicted = outputs.argmax(dim=1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    print(report)
    
    with open(model_name+".txt", "w") as file:
        # Write the text to the file
        file.write(report)
        
        
    # Plotting the classification report
    cm = confusion_matrix(all_labels, all_predictions)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix of " + model_name)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Adding labels to the plot
    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
            
    plt.savefig(model_name+'_Confusion Matrix.png', bbox_inches='tight')
    plt.show()




def result_plot2(model_name:str,
             plot_desc:str,
             data_1:list,
             data_2:list,
             data_3:list,
             DPI=100,
             axis_label_size=15,
             x_grid=0.1,
             y_grid=0.5,
             axis_size=12):
    
    assert(len(data_1)==len(data_2))

    
    fig , ax = plt.subplots(1,figsize=(10,5) , dpi=DPI)

    fig.suptitle(f"Train , validation and Test "+plot_desc,y=0.95 , fontsize=20)

    epochs = range(len(data_1))
    ax.plot(epochs, data_1, 'b',linewidth=3, label=' tarin '+ plot_desc)
    ax.plot(epochs, data_2, 'r',linewidth=3, label=' validation '+plot_desc)
    ax.plot(epochs, data_3, 'g',linewidth=3, label=' test '+plot_desc)

    ax.set_xlabel("epoch"       ,fontsize=axis_label_size)
    ax.set_ylabel(plot_desc     ,fontsize=axis_label_size)
    if x_grid:
        ax.grid(axis="x",alpha=0.1)
    if y_grid:
        ax.grid(axis="y",alpha=0.5)
    

    ax.legend(loc=0,prop={"size":9})

    ax.tick_params(axis="x",labelsize=axis_size)
    ax.tick_params(axis="y",labelsize=axis_size)

    #spine are borde line of plot
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # ax.set_ylim([0.0,0.25])
    plt.savefig(model_name+'.png', bbox_inches='tight')
    plt.show()

def result_plot(model_name:str,
             plot_desc:str,
             data_1:list,
             data_2:list,
             DPI=100,
             axis_label_size=15,
             x_grid=0.1,
             y_grid=0.5,
             axis_size=12):
    
    assert(len(data_1)==len(data_2))
    
    '''
        in this function by getting the two list of data result will be ploted as the 
        TA desired
        
        Parameters
        ----------
        model_name:str : dosent do any thing in this vesion but can implemented to save image file
            with the given name
        
        plot_desc:str : define the identity of the data i.e. Accuracy , loss ,...
        
        data_1:list     : list of first data set .
        data_2:list     : list of second data set .
        
        optional
        
        DPI=100             :   define the quality of the plot
        axis_label_size=15  :   define the label size of the axises
        x_grid=0.1          :   x_grid capacity
        y_grid=0.5          :   y_grid capacity
        axis_size=12        :   axis number's size
        
        
        Returns
        -------
        none      : by now 
        

        See Also
        --------
        size of the two list must be same 

        Notes
        -----
        size of the two list must be same 

        Examples
        --------
        >>> result_plot("perceptron",
             "loss",
             data_1,
             data_2)

        '''
    
    fig , ax = plt.subplots(1,figsize=(10,5) , dpi=DPI)

    fig.suptitle(f"Train and validation "+plot_desc,y=0.95 , fontsize=20)

    epochs = range(len(data_1))
    ax.plot(epochs, data_1, 'b',linewidth=3, label='tarin '+ plot_desc)
    ax.plot(epochs, data_2, 'r',linewidth=3, label='validation '+plot_desc)

    ax.set_xlabel("epoch"       ,fontsize=axis_label_size)
    ax.set_ylabel(plot_desc     ,fontsize=axis_label_size)
    if x_grid:
        ax.grid(axis="x",alpha=0.1)
    if y_grid:
        ax.grid(axis="y",alpha=0.5)
    

    ax.legend(loc=0,prop={"size":9})

    ax.tick_params(axis="x",labelsize=axis_size)
    ax.tick_params(axis="y",labelsize=axis_size)

    #spine are borde line of plot
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # ax.set_ylim([8.48,8.6])
    plt.savefig(model_name+'.png', bbox_inches='tight')
    plt.show()
    


def predict_image(img:torch.tensor,
                  model:torch.nn.Module,
                  classes:list,
                  device='cpu'):
    with torch.no_grad():
        # Convert to a batch of 1
        xb = img.unsqueeze(0).to(device)
        # Get predictions from model
        yb = model(xb)
        # Pick index with highest probability
        _, preds  = torch.max(yb, dim=1)
        # Retrieve the class label
        return classes[preds[0].item()]
    
def prediction(model_name:str,
               classes:list,
               model:torch.nn.Module,
               teste_data_sets):
    
    randd=torch.randint(0,10000,[3])

    #####################################

    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(10,5))

    fig.suptitle(model_name+' some predicton', fontsize=20)

    img0, label0 = teste_data_sets[randd[0]]
    ax1.set_title('Label:'+classes[label0]+', Predicted:'+predict_image(img0, model,classes))
    ax1.imshow(img0.permute(1, 2, 0))

    img1, label1 = teste_data_sets[randd[1]]
    ax2.set_title('Label:'+classes[label1]+', Predicted:'+predict_image(img1, model,classes))
    ax2.imshow(img1.permute(1, 2, 0))

    img2, label2 = teste_data_sets[randd[2]]
    ax3.set_title('Label:'+classes[label2]+', Predicted:'+predict_image(img2, model,classes))
    ax3.imshow(img2.permute(1, 2, 0))



    plt.tight_layout()
    # Make space for title
    plt.subplots_adjust(top=0.95)
    plt.savefig(model_name+'.png', bbox_inches='tight')
    plt.show()