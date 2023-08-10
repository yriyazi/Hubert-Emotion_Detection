import torch
import librosa
import numpy as np

def preprocess_audio(file_path,K = 20):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=16000)
    # Ensure consistent duration (10 seconds in this example)
    audio = librosa.util.fix_length(data = audio,size = 16000 * 6)
    # Extract mel spectrogram features
    mel_spec = librosa.feature.melspectrogram(y = audio, sr=sr, n_fft=1024, hop_length=512)
    # Convert to log-mel spectrogram
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    # Normalize the spectrogram
    norm_mel_spec = (log_mel_spec - log_mel_spec.mean()) / log_mel_spec.std()
    # Convert to PyTorch tensor
    mel_tensor = torch.from_numpy(norm_mel_spec).float()
    # Normalize the data
    data_mean       = torch.mean(mel_tensor, dim=0)
    data_std        = torch.std(mel_tensor, dim=0)
    normalized_data = (mel_tensor - data_mean) / data_std
    covariance_matrix = torch.matmul(mel_tensor.T, mel_tensor) / (mel_tensor.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
        # Number of principal components to keep
    selected_eigenvectors = eigenvectors[:, -K:]  # Select the last k eigenvectors
    transformed_data = torch.matmul(normalized_data, selected_eigenvectors)

    return mel_tensor.unsqueeze(0)