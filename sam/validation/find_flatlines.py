import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
import logging
logger = logging.getLogger(__name__)


class RemoveFlatlines(BaseEstimator, TransformerMixin):
    '''
    Detect flatlines and set to nan. Note that you have to check whether
    signals can contain natural flatliners (such as machines turned off),
    that might not need to be removed.

    Parameters
    ----------
    cols: list of strings (defaults to None)
        columns to apply this method to. If None, will apply to every column.
    window: int (default = 1)
        number of previous equal values to consider current value a flatliner.
        so if set to 2, requires that 2 previous values are identical to
        current to set current value to nan.

    Examples
    --------
    >>> from sam.validation import RemoveFlatlines
    >>> # create some data
    >>> data = [1, 2, 6, 3, 4, 4, 4, 3, 6, 7, 7, 2, 2]
    >>> # with one clear outlier
    >>> test_df = pd.DataFrame()
    >>> test_df['values'] = data
    >>> # now detect extremes
    >>> cols_to_check = ['values']
    >>> RF = RemoveFlatlines(
    >>>     cols=cols_to_check,
    >>>     window=2)
    >>> data_corrected = RF.fit_transform(test_df)
    >>> fig = diagnostic_flatline_removal(RF, test_df, 'values')
    '''

    def __init__(self, cols=None, window=1):

        self.cols = cols
        self.window = window

    def _search_sequence_numpy(self, arr, seq):
        """
        from: https://stackoverflow.com/questions/36522220/
              searching-a-sequence-in-a-numpy-array
        This function returns the indices in arr that match seq.
        For example, with arr = [2, 0, 0, 1, 0, 1, 0, 0] and seq = [0, 0],
        this returns [1, 2, 6, 7].

        Parameters
        ----------
        arr: np.array or list of scalars
            the array you want to search
        seq: np.array or list of scalars
            the sequence you want to find

        Output
        ------
        Output : 1D Array of indices in the input array that satisfy the
        matching of input sequence in the input array.
        In case of no match, an empty list is returned.
        """

        # cast to np.array (changes nothing if they already were)
        arr = np.array(arr)
        seq = np.array(seq)

        # Store sizes of input array and sequence
        Na, Nseq = arr.size, seq.size

        # Range of sequence
        r_seq = np.arange(Nseq)

        # Create a 2D array of sliding indices across the entire length
        # of input array. Match up with the input sequence & get
        # the matching starting indices.
        M = (arr[np.arange(Na-Nseq+1)[:, None] + r_seq] == seq).all(1)

        # Get the range of those indices as final output
        if M.any() > 0:
            return np.where(np.convolve(M, np.ones((Nseq), dtype=int)) > 0)[0]
        else:
            return np.array([])  # No match found

    def fit(self, data):
        return self

    def transform(self, data):
        '''
        Parameters
        ----------
        data: pandas dataframe
            with index as increasing time and columns as features

        Returns:
        -------
        data_r: pandas dataframe
            with flatlines replaced by nans
        '''

        self.invalids = {}
        data_r = data.copy()

        for col in self.cols:

            these_data = data.loc[:, col]

            # to start, use window of 1
            flatliners = np.array((these_data.shift(1)-these_data) == 0)

            # now see if they expand across the self.window n samples
            for w in range(2, self.window+1):
                flatliners &= np.array((these_data.shift(w)-these_data) == 0)

            # prepend all nans with window amount of nans
            seq = np.hstack([np.zeros(self.window), [1]])
            indices = self._search_sequence_numpy(
                flatliners.astype(int), seq)

            if len(indices) > 0:
                flatliners[indices] = True

            # save to self for later plot
            self.invalids[col] = flatliners

            logger.info(
                'detected %d ' % np.sum(flatliners) +
                'flatline samples in %s ' % col +
                'with window of %d ' % self.window)

            # now replace with nans
            data_r[col].iloc[flatliners] = np.nan

        return data_r
