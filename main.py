import librosa
import soundfile
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype='float32')
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result


# DataFlair - Emotions in the RAVDESS data_set
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# DataFlair - Emotions to observe
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']


# DataFlair - Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x, y = [], []
    for file in glob.glob("ML_dataset\\Actor_*\\*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


def main():
    print('hello< CNN FIT >\nPlease wait </>')

    # DataFlair - Split the data_set
    x_train, x_test, y_train, y_test = load_data(test_size=0.25)

    # DataFlair - Get the shape of the training and testing data_sets
    print((x_train.shape[0], x_test.shape[0]))

    # DataFlair - Get the number of features extracted
    print("==========")
    print(f'Features extracted: {x_train.shape[1]}')

    # DataFlair - Initialize the Multi Layer Perceptron Classifier
    model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08,
                          hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

    # DataFlair - Train the model
    model.fit(x_train, y_train)

    print("CNN  == End the FIT")

    # DataFlair - Predict for the test set
    y_pred = model.predict(x_test)

    # DataFlair - Calculate the accuracy of our model
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

    # DataFlair - Print the accuracy
    print("=======")
    print("Accuracy: {:.2f}%".format(accuracy*100))







if __name__ == '__main__':
    main()
