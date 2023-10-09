import mirdata
import librosa
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score 
from sklearn.neighbors import KNeighborsClassifier


def load_data(data_home):
    """
    Load the mini-Medley-Solos-DB dataset.

    Parameters
    ----------
    data_home : str
                Path to where the dataset is located

    Returns
    -------
    dataset: mirdata.Dataset
             The mirdata Dataset object correspondong to Medley-Solos-DB
    """
    
    # YOUR CODE HERE
    # Hints: 
    # Look at the mirdata tutorial on how to initialize a dataset.
    # Define the correct path using the data_home argument.
    import mirdata
    dataset = mirdata.initialize('medley_solos_db', data_home=data_home)
    return dataset

def split_data(tracks):
    """
    Splits the provided dataset into training, validation, and test subsets based on the 'subset' 
    attribute of each track.

    Parameters
    ----------
    track_list : list
                 list of dataset.track objects from Medley_solos_DB dataset
    # tracks will be dataset.load_tracks(), so is a dictionary.
    
    Returns
    -------
    tracks_train : list
        List of tracks belonging to the 'training' subset.
    tracks_validate : list
        List of tracks belonging to the 'validation' subset.
    tracks_test : list
        List of tracks belonging to the 'test' subset.
    """
    # YOUR CODE HERE
    import mirdata
    tracks_train = []
    tracks_validate = []
    tracks_test = []

    for _, track_obj in tracks.items(): # Bc dict.items() method returns pairs of (key, value). We want to skip the 'key'.
        if track_obj.subset == 'training':
            tracks_train.append(track_obj)
        elif track_obj.subset == 'validation':
            tracks_validate.append(track_obj)
        elif track_obj.subset == 'test':
            tracks_test.append(track_obj)

    return tracks_train, tracks_validate, tracks_test

def compute_mfccs(y, sr, n_fft=1024, hop_length=512, n_mels=128, n_mfcc=20):
    """
    Compute mfccs for an audio file using librosa, removing the 0th MFCC coefficient.
    
    Parameters
    ----------
    y : np.array
        Mono audio signal
    sr : int
        Audio sample rate
    n_fft : int
        Number of points for computing the fft
    hop_length : int
        Number of samples to advance between frames
    n_mels : int
        Number of mel frequency bands to use
    n_mfcc : int
        Number of mfcc's to compute
    
    Returns
    -------
    mfccs: np.array (t, n_mfcc - 1)
        Matrix of mfccs

    """
    # YOUR CODE HERE
    import librosa
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length, n_mels=n_mels).T[:, 1:] # mfccs.shape = (n_mfcc, t) # Transpose to np.array (t, n_mfcc) and drop the 1st mfcc # each row corresponds to a time frame
    return mfccs



def get_stats(features):
    """
    Compute summary statistics (mean and standard deviation) over a matrix of MFCCs.
    Make sure the statitics are computed across time (i.e. over all examples, 
    compute the mean of each feature).

    Parameters
    ----------
    features: np.array (n_examples, n_features)
              Matrix of features
               
    [[mfcc1(t_0), mfcc2(t_0), ... , mfcc_n_mfcc-1(t_0)],
     [mfcc1(t_1), mfcc2(t_1), ... , mfcc_n_mfcc-1(t_1)],
     ...
     [mfcc1(t_t), mfcc2(t_t), ... , mfcc_n_mfcc-1(t_t)]]


    Returns
    -------
    features_mean: np.array (n_features)
                   The mean of the features
                   
    features_mean = 
    [mean(mfcc1 across all t), mean(mfcc2 across all t), ... , mean(mfcc_n_mfcc-1 across all t)]

    features_std: np.array (n_features)
                   The standard deviation of the features
                   
    features_std = 
    [std(mfcc1 across all t), std(mfcc2 across all t), ... , std(mfcc_n_mfcc-1 across all t)]

    """
    # Hint: use numpy mean and std functions, and watch out for the axis.
    # YOUR CODE HERE
    features_mean = np.mean(features, axis=0) # rows for features(aka mfccs)
    features_std = np.std(features, axis=0)
    return features_mean, features_std

def normalize(features, features_mean, features_std):
    """
    Normalize (standardize) a set of features using the given mean and standard deviation.

    Parameters
    ----------
    features: np.array (n_examples, n_features)
              Matrix of features
    features_mean: np.array (n_features)
              The mean of the features
    features_std: np.array (n_features)
              The standard deviation of the features

    Returns
    -------
    features_norm: np.array (n_examples, n_features)
                   Standardized features

    """

    # YOUR CODE HERE
    features_norm = (features - features_mean) / features_std # broadcasting in NumPy duplicate the mean and std.
    return features_norm



def get_features_and_labels(track_list, n_fft=2048, hop_length=512, n_mels=128, n_mfcc=20):
    """
    Our features are going to be the `mean` and `std` MFCC values of a track concatenated 
    into a single vector of size `2*n_mfcss`. 

    Create a function `get_features_and_labels()` such that extracts the features 
    and labels for all tracks in the dataset, such that for each audio file it obtains a 
    single feature vector. This function should do the following:

    For each track in the collection (e.g. training split),
        1. Compute the MFCCs of the input audio, and remove the first (0th) coeficient.
        2. Compute the summary statistics of the MFCCs over time:
            1. Find the mean and standard deviation for each MFCC feature (2 values for each)
            2. Stack these statistics into single 1-d vector of length ( 2 * (n_mfccs - 1) )
        3. Get the labels. The label of a track can be accessed by calling `track.instrument_id`.
    Return the labels and features as `np.arrays`.

    Parameters
    ----------
    track_list : list
                 list of dataset.track objects from Medley_solos_DB dataset
    n_fft : int
                 Number of points for computing the fft
    hop_length : int
                 Number of samples to advance between frames
    n_mels : int
             Number of mel frequency bands to use
    n_mfcc : int
             Number of mfcc's to compute

    Returns
    -------
    feature_matrix: np.array (len(track_list), 2*(n_mfcc - 1))
        The features for each track, stacked into a matrix.
    label_array: np.array (len(track_list))
        The label for each track, represented as integers
    """

    # Hint: re-use functions from previous parts (e.g. compute_mfcss and get_stats)
    # YOUR CODE HERE
    feature_matrix = []
    label_array = []

    for track in track_list:
        y, sr = librosa.load(track.audio_path, sr=None) # read every track
        mfccs = compute_mfccs(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc) # get the 2D MFCCs array
        mean, std = get_stats(mfccs)
        features = np.hstack((mean, std))
        feature_matrix.append(features)
        label_array.append(track.instrument_id)

    feature_matrix = np.array(feature_matrix)
    label_array = np.array(label_array)

    return feature_matrix, label_array


def fit_knn(train_features, train_labels, validation_features, validation_labels, ks=[1, 5, 10, 50]):
    """
    Fit a k-nearest neighbor classifier and choose the k which maximizes the
    *f-measure* on the validation set.
    
    Plot the f-measure on the validation set as a function of k.

    Parameters
    ----------
    train_features : np.array (n_train_examples, n_features)
        training feature matrix
    train_labels : np.array (n_train_examples)
        training label array
    validation_features : np.array (n_validation_examples, n_features)
        validation feature matrix
    validation_labels : np.array (n_validation_examples)
        validation label array
    ks: list of int
        k values to evaluate using the validation set

    Returns
    -------
    knn_clf : scikit learn classifier
        Trained k-nearest neighbor classifier with the best k
    best_k : int
        The k which gave the best performance
    """
    
    # Hint: for simplicity you can search over k = 1, 5, 10, 50. 
    # Use KNeighborsClassifier from sklearn.
    # YOUR CODE HERE
    
    f_measures = []
    
    for k in [1, 5, 10, 50]:
        
        knn_clf = KNeighborsClassifier(n_neighbors=k)
        knn_clf.fit(train_features, train_labels)
        
        predicted_labels = knn_clf.predict(validation_features)
        f = f1_score(validation_labels, predicted_labels, average='micro')
        f_measures.append(f)
        
    plt.plot(ks, f_measures, '-o')
    plt.xlabel('k')
    plt.ylabel('F-measure')
    plt.title('F-measure vs. k')
    plt.show()
    
    best_index = np.argmax(f_measures)
    best_k = ks[best_index]
    knn_clf = KNeighborsClassifier(n_neighbors=best_k)
    knn_clf.fit(train_features, train_labels)
    
    return knn_clf, best_k
    
