import pywt
from scipy import stats
from scipy.signal import periodogram
from scipy import trapz
from scipy import io
import numpy as np
import math
import itertools as itl

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.svm import SVC

# from sklearn.svm import SVC


def wrcoef(X, coef_type, coeffs, wavename, level):
    N = np.array(X).size
    a, ds = coeffs[0], list(reversed(coeffs[1:]))


    if coef_type =='a':
        return pywt.upcoef('a', a, wavename, level=level)[:N]
    elif coef_type == 'd':
        return pywt.upcoef('d', ds[level-1], wavename, level=level)[:N]
    else:
        raise ValueError("Invalid coefficient type: {}".format(coef_type))


def bandpower(x, fs, fmin, fmax):
    f, Pxx = scipy.signal.periodogram(x, fs=fs)
    ind_min = scipy.argmax(f > fmin) - 1
    ind_max = scipy.argmax(f > fmax) - 1
    return scipy.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])


def wavelets_features(target):
    n_chan = np.size(target, 1)
    n_subj = np.size(target, 2)
    n_feat = 10
    features = np.zeros((n_subj, n_chan, n_feat*3))
    for s in range(n_subj):
        for c in range(n_chan):
            signal = target[:, c, s]
            cl = pywt.wavedec(signal, 'db4', level=5)
            total = []
            total += extract_feat(wrcoef(signal,'d', cl, 'db4', 4))
            total += extract_feat(wrcoef(signal,'d', cl, 'db4', 5))
            total += extract_feat(wrcoef(signal,'a', cl, 'db4', 5))
            features[s, c, :] = total 
    return features


def extract_feat(signal):
    RMS = np.sqrt(np.mean(signal**2))   # Root Mean Square (RMS) power of the signal
    MAV = np.mean(np.abs(signal))       # Mean (MEAN), first order mode
    IEEG = np.sum(np.abs(signal))       # integrated EEG (IEEG)
    SSI = np.sum(np.abs(signal)**2)     # Simple Square Integral (SSI)
    VAR = np.var(signal)                # Variance (VAR) second order moment
    signal_shift = signal                       
    signal_shift[0] = 0
    signal_out = signal
    signal_out[len(signal)-1] = 0      # next - previous
    AAC = np.mean(np.abs(signal_shift - signal_out))    # Average Amplitude Change (ACC)
    SKV = stats.skew(signal)            # Skewness (SKEW) third order moment
    KURT = stats.kurtosis(signal)       # Kurtosis (KURT) fourth order moment
    ENT = stats.entropy(signal)         # Shannon Entropy (ENTR), randomness of signal
    _, pxx = periodogram(signal, fs=256, nfft=1024)
    BP = trapz(pxx)                     # average power in the input signal vector
    return [RMS, MAV, IEEG, SSI, VAR, AAC, SKV, KURT, ENT, BP]

def flatten_features(features):
    res = [feat.flatten() for feat in features ]

    return np.nan_to_num(res)


def cross_validation(data, target):
    params ={ 'C': [0.1, 0.5, 1, 5, 10, 20, 100, 500, 1000]}
    # gamma = np.arange(0.1, 15, 0.1)
    clf = GridSearchCV(SVC(), params, cv=10)
    clf.fit(data, target)
    return clf.best_score_, clf.best_params_
 

def learn(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(flatten_features(features), labels[0])
    score, params = cross_validation(X_train, y_train)
    print(score)
    clf = SVC(**params)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    acc_score = accuracy_score(y_test, predict)
    print(acc_score)
    return acc_score

data = io.loadmat('visual_oddball_p300_FzCzPz.mat')
target = data['Target']
features = wavelets_features(target)
labels = data['subject_labels']
res = learn(features, labels)
print(res)
# wavelets_features()