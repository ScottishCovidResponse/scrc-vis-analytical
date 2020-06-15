## Correlation metrics


In this prototype, we look at time series and how they relate to each other.
We focus in particular on similarity metrics such as
- zero-mean normalised cross-correlation,
- Pearson product-moment correlation coefficient,
- Spearman rank-order correlation coefficient, and
- Kendall's tau is a measure of the correspondence between two rankings.


### Technology

This protoype uses Python with numpy, pandas and scipy libraries (required),
and dash and plotly express (optional) for displaying the results.

For setting up, you need to download XXX and run the following code, e.g.:

```
pip3 install --user numpy pandas scipy dash plotly matplotlib
```

### List of files

- `utils.py` implements the core of this prototype. It relies on the Numpy, Pandas and Scipy Libs.
  - `loadDataFiles()`: Load the time series from the CSV files to the main memory.
  - `getTimeSeries(file: str, column: str, window: str) -> pd.DataFrame`: Extract a time series (`column`) from a given file (`file`). If required, the data is smoothed using a filter (`window`).
  - `getSimilarityMetrics(file1, file2, filter) -> pd.DataFrame`: Compute the similarity metrics between the time series contained in two data
  files (`file1` and `file2`). Note that `file1` and `file2` may be equal.
  If required, the data is smoothed using a filter (`window`).
  Similarity metrics considered here are:
      - zero-mean normalised cross-correlation ('ZNCC'),
      - Pearson product-moment correlation coefficient ('pearsonr'),
      - Spearman rank-order correlation coefficient ('spearmanr'),
      - Kendall's tau is a measure of the correspondence between two rankings.
  - `correlateTimeSeries(series1, series2) -> np.array`: Compute the cross-correlation between two 1-D signals (e.g. time series).
  - `getZeroMeanNormalised(series: np.array) -> np.array`: returns a 1-D signal after zero-mean and unit-variance normalisation
  - `removeComma(intAsStr: str) -> int`: Remove the comma from an int stored as a string.
  - `removeMissingData(series) -> np.array`:
      Remove samples that are equal to `*` in `series` and make sure the commas have been removed from numbers. Note that this function will change in the future.
  - `getZNCC(series1, series2) -> float:` Returns the zero-mean normalised cross-correlation ('ZNCC') between two signals.
  - `smooth(x,window_len=11,window='hanning') -> np.array`: Smooth the data using a window with requested size. This function comes from [https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html](https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html).
- `vis-corr-app.py`: Is the interactive visualisation protoypes to test functionalities implemented in `utils.py`. It obviously relies on `utils.py`, but also on the Dash and Plotly Express Libs.
- `ICU-patients.csv`, `cumulative-cases.csv`, `hospital_confirmed.csv`, and `hospital_suspected.csv`: Input datasets used for testing.


### Example screenshots

Here is a screenshot from Correlation metrics protoype. A correlation matirx is computed (see left). When we click on a dot in the matrix, the corresponding time series are shown (right). Here the time series are not normalised:
![Screen-1](/images/ZNCC-app-not_normalised.png)

Same as above, but here the time series are normalised using zero-mean and unit-variance normalisation:
![Screen-2](/images/ZNCC-app-normalised.png)

Here is a screenshot from Correlation metrics protoype to show how cross-correaltion can be used to estimate a lag between two time series. In the next two screenshots the bottom graph show the cross-correlation function between the two time series. The middle graph plots the four times series after one of them has been corrected by taking into accounts the lag.

- Here there is no lag between the two series:
  - Confirmed cases in NHS Dumfries & Galloway vs. NHS Tayside:
  ![Screen-3](/images/correction_of_time_sereis_with_cross-correlation3.png)
- In the next three screenshots, there is a lag.
  - confirmed cases in NHS Lothian vs. NHS Tayside:
  ![Screen-4](/images/correction_of_time_sereis_with_cross-correlation1.png)
  - confirmed cases in NHS Forth Valley vs. NHS Lothian:
  ![Screen-5](/images/correction_of_time_sereis_with_cross-correlation2.png)
  - ICU patients in NHS Ayshire & Arran vs. confirmed cases in NHS Glasgow & Glyde:
  ![Screen-6](/images/correction_of_time_sereis_with_cross-correlation4.png)
