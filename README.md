# Huber-Emotion_Detection


![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)

## Project Description

Huber-Emotion_Detection is a project that utilizes the Hugging Face Transformers library to perform audio emotion detection. The main objective of this project is to classify audio emotions into six different categories using the "facebook/hubert-base-ls960" model. The dataset used for training and evaluation is the [Shemo Persian Speech Emotion Detection Database](https://www.kaggle.com/datasets/mansourehk/shemo-persian-speech-emotion-detection-database).

## Installation

Make sure you have Python installed, then install the required packages using the following command:

```bash
pip install transformers
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/Huber-Emotion_Detection.git
cd Huber-Emotion_Detection
```

2. Download the dataset from [here](https://www.kaggle.com/datasets/mansourehk/shemo-persian-speech-emotion-detection-database) and place it in the appropriate directory.

3. Run the scripts to preprocess the data and train the model:

```bash
python preprocess.py
python train.py
```

4. Use the trained model for emotion detection:

```python
# Load the trained model
from transformers import HubertForSequenceClassification, HubertTokenizer

model_name = "facebook/hubert-base-ls960"
model = HubertForSequenceClassification.from_pretrained(model_name)
tokenizer = HubertTokenizer.from_pretrained(model_name)

# Load an audio file and convert it to text
audio_text = "..."
input_ids = tokenizer(audio_text, return_tensors="pt").input_ids

# Perform emotion classification
with torch.no_grad():
    outputs = model(input_ids)
    predicted_class = torch.argmax(outputs.logits).item()
```

## Training Plots

<p float="center">
  <img src="Pictures/Hubert_Hiddenstates_Accuracy.png" width="400" />
  <img src="Pictures/Hubert_Hiddenstates_loss.png" width="400" /> 
</p>
<p align="center" width="100%">
    <img width="90%" src="Pictures/Hubert_Hiddenstates_Confusion Matrix.png"> 
</p>

## Contribution Guidelines

Contributions to the project are welcome! If you want to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your fork.
5. Open a pull request, describing the changes you've made.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

Remember to replace placeholders like `your-username` with your actual GitHub username, and provide appropriate paths and filenames for the preprocessing and training scripts as needed. Also, make sure to have the necessary imports and configurations in your actual project files.