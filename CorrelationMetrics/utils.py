# -*- coding: utf-8 -*-


"""A file that includes functions useful to analyse time series.
It includes pre-processing functions (e.g. zero-mean and unit
variance normalistion) and functions performing the analysis
(e.g. correlations).
"""

__author__ = "Dr Franck P. Vidal";
__copyright__ = "Copyright 2007, The Cogent Project";
__credits__ = ["Dr Franck P. Vidal"];

__maintainer__ = "Dr Franck P. Vidal"
__email__ = "f.vidal@bangor.ac.uk"
__status__ = "Development"


# Built-in/Generic Imports
import math

# Libs
import numpy as np;
import pandas as pd;

import scipy
from scipy import stats as stats
from scipy import signal as signal
from skimage import metrics as ski_metrics
from sklearn import metrics as skl_metrics


window_set = ['none', 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'];
""" List of filters that can be used to smooth a 1-D signal (here time series). """

data_files = ["cumulative-cases.csv", "hospital_confirmed.csv", "hospital_suspected.csv", "ICU-patients.csv"];
""" List of files that contain input data (here time series). """

similarity_metrics_set = ["ZNCC", "pearsonr", "spearmanr", "kendalltau", "SSIM", "PSNR", "MSE", "NRMSE", "ME", "MAE", "MSLE", "MedAE"];
""" List of similarity and dissimilarity metrics (.e.g correlations, distance errors) that can be used. """

NoneType = type(None);
""" The NoneType/ """

input_df_set = None;
""" A dictionary that contains the Pandas DataFrames for each of the files in data_files.
The keys correspond to the file names in data_files. """


def loadDataFiles():
    """ Load the time series in data_files.

    Return input_df_set, the dictionary that contains the Pandas DataFrames for each of the files in data_files.
    The keys correspond to the file names in data_files.
    """

    global input_df_set;

    # No need to load the data if it is already in memory
    if isinstance(input_df_set, NoneType):
        input_df_set = {};

        for file in data_files:
            input_df_set[file] = pd.read_csv(file);

            # Convert the 'Date' cell to the Pandas datetime data type.
            input_df_set[file]['Date'] = pd.to_datetime(input_df_set[file]['Date'],format='%d/%m/%Y');

    return input_df_set;


def correlateTimeSeries(series1, series2) -> np.array:
    """ Compute the cross-correlation between two 1-D signals (e.g. time series).

    It quantifies the level of similarity between two series as a function of the displacement of one relative to the other.
    If the inputs are both normalised, the cross correlation values are in the [-1, 1] range.
    -1 corresponds to a strong anticorrelation, or inverse correlation,;
    0 corresponds to a non-correlation; and
    1 corresponds to a strong correlation.

    Returns the np.array that contains the cross correlation.
    """

    # Use numpy to compute the cross correlation
    cross_corr = np.correlate(np.array(series1, dtype=np.float32), np.array(series2, dtype=np.float32), "same");
    return cross_corr / len(series1);

    # Use numpy to compute the cross correlation
    return scipy.correlate(np.array(series1), np.array(series2), mode='same');


def getTimeSeries(file: str, column: str, window: str) -> pd.DataFrame:
    """ Extract a time series (column) from a given file (file). If required, the data is smoothed using a filter (window).

    Return the time series as a Pandas DataFrame
    """

    global input_df_set;

    # Load the data if needed
    loadDataFiles();

    # Create the column headers of the DataFrame
    columns = ["Date", "Count", "Normalised count", "data"];

    # Create the DataFrame
    output_df = pd.DataFrame(columns=columns);

    # Make sure the the inputs are valid (may be required due to the GUI)
    if not isinstance(file, NoneType) and not isinstance(column, NoneType) and not isinstance(window, NoneType):

        # Create the file name
        filename = file + ".csv";

        # Ignore missing data
        selected_rows = input_df_set[filename][column] != "*";
        temp_X = input_df_set[filename][selected_rows]['Date'];
        temp_Y = input_df_set[filename][selected_rows][column];

        X = [];
        Y1 = [];

        for x, y in zip(temp_X, temp_Y):
            if y != '*':
                X.append(x);

                # Remove the commas from the data and convert it to int if necessary
                if isinstance(y, str):
                    Y1.append(int(y.replace(',', '')));
                else:
                    Y1.append(y);

        # Filter the data if necessary
        if window != 'none':
            Y1 = smooth(x=np.array(Y1),window_len=5,window=window);

        # Normalise the data
        Y2 = Y1 - np.average(Y1);
        Y2 /= np.std(Y1);

        # Add all the rows to the DataFrame
        for x, y1, y2 in zip(X, Y1, Y2):
            new_row = {
                "Date": x,
                "Count": y1,
                "Normalised count": y2,
                "data": file + "/" + column
            };

            output_df = output_df.append(new_row, ignore_index=True)

    return output_df;


def getSimilarityMetrics(file1, file2, window) -> pd.DataFrame:
    """Compute the similarity metrics between the time series contained in two data
    files (file1 and file2). Note that file1 and file2 may be equal.
    If required, the data is smoothed using a filter (window).

    Similarity metrics considered here are:
    - zero-mean normalised cross-correlation ('ZNCC'),
    - Pearson product-moment correlation coefficient ('pearsonr'),
    - Spearman rank-order correlation coefficient ('spearmanr'),
    - Kendall's tau is a measure of the correspondence between two rankings ('kendalltau').
    - Structural similarity ('SSIM').
    - Peak signal-to-noise ratio is a ratio between singal and noise ('PSNR').

    Dissimilarity metrics considered here are:
    - Mean squared error ('MSE').
    - Normalised root mean squred error ('NRMSE').
    - Max error ('ME')
    - Mean absolute error ('MAE').
    - Mean squared log error ('MLSE').
    - Median absolute error ('MedAE').

    For 'ZNCC', 'pearsonr', 'spearmanr', and 'SSIM' values are in the [-1, 1] range.
    -1 corresponds to a strong anticorrelation, or inverse correlation;
    0 corresponds to a non-correlation; and
    1 corresponds to a strong correlation.

    For 'kendalltau', values close to 1 indicate strong agreement, values close to -1 indicate strong disagreement.

    For 'PSNR', high value usually represents low noise level, low value usually indicates high noise.

    For 'MSE', 'NRMSE', 'ME', 'MAE', 'MSLE', and 'MedAE', 0.0 indicates two series are identical. Higher value indicates lower correlation."""

    # Create the column headers of the DataFrame
    columns = ['x', 'y'];
    for col in similarity_metrics_set:
        columns.append(col);

    # Create the DataFrame
    output_df = pd.DataFrame(columns = columns);


    # Make sure the the inputs are valid (may be required due to the GUI)
    if not isinstance(file1, NoneType) and not isinstance(file2, NoneType) and not isinstance(window, NoneType):

        global input_df_set;

        # Load the data if needed
        loadDataFiles();

        # Create file filenames
        filename1 = file1 + ".csv";
        filename2 = file2 + ".csv";

        # Get the corresponding DataFrames
        df1 = input_df_set[filename1];
        df2 = input_df_set[filename2];

        # Process the columns of both DataFrames
        number_of_columns1 = len(df1.columns);
        number_of_columns2 = len(df2.columns);

        # using nested for loops to create a 2-D matrix
        for i in range(number_of_columns1 - 1):
            for j in range(number_of_columns2 - 1):

                X = df1.columns[i + 1];
                Y = df2.columns[j + 1];

                # Remove missing data if needed
                seriesX = removeMissingData(df1[X]);
                seriesY = removeMissingData(df2[Y]);

                # Smooth the data if needed
                if window != 'none':
                    seriesX = smooth(x=seriesX,window_len=5, window=window);
                    seriesY = smooth(x=seriesY,window_len=5, window=window);

                if seriesX.shape != seriesY.shape:
                    raise (ValueError, "Only accepts signals that have the same shape.")

                # Compute the similarity metrics
                zncc = getZNCC(seriesX, seriesY);
                ssim = ski_metrics.structural_similarity(seriesX, seriesY);
                psnr = getPSNR(seriesX, seriesY);

                # Use pandas corr to ingnore inf and nan.
                seriesX, seriesY = pd.Series(seriesX), pd.Series(seriesY),

                pearsonr = seriesX.corr(seriesY, method='pearson');
                spearmanr = seriesX.corr(seriesY, method='spearman');
                kendalltau = seriesX.corr(seriesY, method='kendall');

                # Compute dissimilarity metrics
                mse = ski_metrics.mean_squared_error(seriesX, seriesY);
                nrmse = ski_metrics.normalized_root_mse(seriesX, seriesY);
                me = getME(seriesX, seriesY);
                mae = getMAE(seriesX, seriesY);
                msle = getMSLE(seriesX, seriesY);
                medae = getMedAE(seriesX, seriesY);

                # Add the values to the matrix
                new_row = {
                    'x': X,
                    'y': Y,
                    'ZNCC': zncc,
                    'pearsonr': pearsonr,
                    'spearmanr': spearmanr,
                    'kendalltau': kendalltau,
                    'SSIM': ssim,
                    'PSNR': psnr,
                    'MSE': mse,
                    'NRMSE': nrmse,
                    'ME': me,
                    'MAE': mae,
                    'MSLE': msle,
                    'MedAE': medae
                };

                output_df = output_df.append(new_row, ignore_index=True)

        output_df.to_csv("zncc-" + file1 + "-" + file2 + "-" + window + ".csv");

    return output_df;


def getZeroMeanNormalised(series: np.array) -> np.array:
    """ Zero-mean and unit-variance normalisation.

    Return the normalised 1-D signal."""
    return (series - np.mean(series)) / np.std(series);


def removeComma(intAsStr: str) -> int:
    """ Remove the comma from an int stored as a string.

    Return the corresponding int."""

    return int(intAsStr.replace(',', ''));


def removeMissingData(series) -> np.array:
    """Remove samples that are equal to '*' and make sure the commas have been removed from numbers.

    Returns the two time series without the missing data.
    """

    temp = [];

    for i in series:
        if i == '*':
            temp.append(0);
        elif isinstance(i, str):
            temp.append(removeComma(i));
        elif np.isnan(i) == True:
            temp.append(0);
        elif np.isinf(i) == True:
            temp.append(0);
        else:
            temp.append(i);

    return np.array(temp);


def getZNCC(series1, series2) -> float:
    """ Returns the zero-mean normalised cross-correlation ('ZNCC') between two signals."""

    if series1.shape != series2.shape:
        raise (ValueError, "getZNCC only accepts signals that have the same shape.")

    temp1 = removeMissingData(series1);
    temp2 = removeMissingData(series2);

    return np.mean(np.multiply(getZeroMeanNormalised(temp1), getZeroMeanNormalised(temp2)));


# From https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth(x,window_len=11,window='hanning') -> np.array:
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    # print(x.ndim, x.size, window_len)

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        return x
        raise (ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def getPSNR(series1, series2) -> float:
    """Return 0 if mean squared error between two series is 0.
    PSNR cannot be computed if mean squared error between two series is 0."""

    temp1 = removeMissingData(series1);
    temp2 = removeMissingData(series2);

    data_range = max(max(series1),max(series2));

    if ski_metrics.mean_squared_error(temp1, temp2) == 0.:
        return 0.

    return ski_metrics.peak_signal_noise_ratio(temp1, temp2, data_range=data_range)

def getME(series1, series2) -> float:
    """Returns Max Error (ME) between two signals. Returns 0 if contains NaN or Inf."""

    temp1 = removeMissingData(series1);
    temp2 = removeMissingData(series2);

    return skl_metrics.max_error(temp1, temp2)

def getMAE(series1, series2) -> float:
    """Returns Mean Absolute Error (MAE) between two signals. Returns 0 if contains NaN or Inf."""

    temp1 = removeMissingData(series1);
    temp2 = removeMissingData(series2);

    return skl_metrics.mean_absolute_error(temp1, temp2)

def getMSLE(series1, series2) -> float:
    """Returns Mean Squared Log Error (MSLE) between two signals. Returns 0 if contains NaN or Inf."""

    temp1 = removeMissingData(series1);
    temp2 = removeMissingData(series2);

    return skl_metrics.mean_squared_log_error(temp1, temp2)

def getMedAE(series1, series2) -> float:
    """Returns Median Absolute Error (MedAE) between two signals. Returns 0 if contains NaN or Inf."""

    temp1 = removeMissingData(series1);
    temp2 = removeMissingData(series2);

    return skl_metrics.median_absolute_error(temp1, temp2)
