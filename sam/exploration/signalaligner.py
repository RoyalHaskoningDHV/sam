import numpy as np
import pandas as pd


class SignalAligner:
    """Allows temporal alignment of two signals or dataframes containing two signals
    to be aligned. We assume both signals have the same sampling frequency. For now
    there is no timestamp based alignment, we simply use the cross-correlation of the
    to be aligned signals (thereby assuming equal sampling frequencies).

    Parameters
    ----------
    signal_one : np.ndarray (default=None)
    signal_two : np.ndarray (default=None)

    Examples
    --------
    # Example 1
    >>> import numpy as np
    >>> from sam.exploration.signalaligner import SignalAligner

    >>> signal_one = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0])
    >>> signal_two = np.array([0, 0, 0, 0, 0, 1, 2, 3, 4, 3, 2])
    >>> offset, _ = SignalAligner.align_signals(signal_one, signal_two)

    >>> print('Offset =', offset)

    # Example 2
    >>> N1 = 20
    >>> N2 = 30
    >>> N_aligned = 15

    >>> lat = np.random.randn(N_aligned) + np.random.standard_normal(N_aligned)

    >>> lat1 = np.random.randn(N1)
    >>> i1 = np.random.randint(N1 - N_aligned)
    >>> lat1[i1: i1 + N_aligned] = lat

    >>> lat2 = np.random.randn(N2)
    >>> i2 = np.random.randint(N2 - N_aligned)
    >>> lat2[i2: i2 + N_aligned] = lat

    >>> df1 = pd.DataFrame({'data': np.random.randn(N1), 'lat': lat1})
    >>> df2 = pd.DataFrame({'data': np.random.randn(N2), 'lat': lat2})

    >>> sa = SignalAligner()
    >>> df, offset = sa.align_dataframes(df1, df2, col1, col2, reference=0)
    """
    def __init__(self, signal_one=None, signal_two=None):

        self.signal_one = signal_one
        self.signal_two = signal_two

        if (signal_one is not None) and (signal_two is not None):
            signal_one, signal_two = self._preprocess_signals(signal_one, signal_two)
            self.offset, self.aligned_signal = self.align_signals(signal_one, signal_two)

    def _preprocess_signals(self, signal_one, signal_two):
        """Pad signals to have equal length.

        NOTE: Assumes we have the same sampling frequency for both signals

        Parameters
        ----------
        signal_one : np.ndarray
        signal_two : np.ndarray

        Returns
        -------
        signal_one_pp : np.ndarray
        signal_two_pp : np.ndarray
        """
        N1 = len(signal_one)
        N2 = len(signal_two)
        if N1 < N2:
            signal_one = self._zeropad(signal_one, N2)
            self.signal_one = signal_one
        elif N2 < N1:
            signal_two = self._zeropad(signal_two, N1)
            self.signal_two = signal_two

        # we cannot allow nans for alignment.
        signal_one = np.nan_to_num(signal_one, nan=0.0)
        signal_two = np.nan_to_num(signal_two, nan=0.0)

        return signal_one, signal_two

    def _zeropad(self, signal, N):
        """Zeropad signal to obtain N samples in total.
        We pad the signal at the beginning.

        Parameters
        ----------
        signal : np.ndarray
            Input signals
        N : int
            Number of samples to pad to

        Returns
        -------
        signal_pad : np.ndarray
            Zero-padded input signal
        """
        signal_pad = np.concatenate([np.zeros((N - len(signal), )), signal])
        return signal_pad

    @staticmethod
    def align_signals(signal_one, signal_two):
        """Assuming we have two signals that measure the same variable,
        are at the same sampling frequency, but are misaligned in time,
        return the aligned signal and fill non-overlapping entries with
        np.nan. Signals are assumed to already have the same length.

        Parameters
        ----------
        signal_one : np.ndarray
        signal_two : np.ndarray

        Returns
        -------
        offset : int
            The number samples of misalignment between the two signals.
            A negative offset means signal_one is lagging behind signal_two.
            A positive offset means signal_two is lagging behind signal_one.
        aligned_signal : np.ndarray
            Aligned signal, non-overlapping values are filled with np.nan
        """
        xcorr = np.correlate(signal_one, signal_two, mode='full')
        offset = len(xcorr) // 2 - np.where(xcorr == np.amax(xcorr))[0][0]

        aligned_signal = np.full(signal_one.shape, np.nan)
        if offset < 0:
            aligned_signal[:offset] = signal_two[:offset]
        else:
            aligned_signal[:-offset] = signal_one[:-offset]

        return offset, aligned_signal

    def _pad_df_with_nans(self, df, offset):
        """Pad a dataframe with nan-filled rows either at the
        beginning or the end (depending on the offset).

        Parameters
        ----------
        df : pd.DataFrame
        offset : int
            Number of rows to pad (always takes the absolute value). When
            negative append nans at the beginning, otherwise append at the end.
            The direction is completely arbitrary and unnecssary.
        """
        N_col = df.shape[1]

        nan_array = np.zeros((abs(offset), N_col))
        nan_array[:] = np.nan

        df_nan = pd.DataFrame(nan_array)
        df_nan.columns = df.columns

        if offset < 0:
            return pd.concat([df_nan, df], axis=0)
        else:
            return pd.concat([df, df_nan], axis=0)

    def align_dataframes(self, df1, df2, col1, col2, reference=None):
        """Instead of just aligning two numpy array signals, we might want to
        align two pandas data frames based on specific columns.

        Parameters
        ----------
        df1 : pd.DataFrame
        df2 : pd.DataFrame
        col1 : str
            Column to align with col2 from df2
        col2 : str
            Column to align with col1 from df1
        reference : int (default=None)
            When not None, we return the output relative to the first (reference=0) or
            second (reference=1) dataframe input, such that the shape is preserved and
            data unchaged.

        Returns
        -------
        df_aligned : pd.DataFrame
            combined dataframes. We append np.nan for aligning rows.
        offset : int
            Number of rows to pad (always takes the absolute value). When
            negative append nans at the beginning, otherwise append at the end.
            The direction is completely arbitrary and unnecssary.
        """
        row_diff = df1.shape[0] - df2.shape[0]
        if row_diff < 0:
            df1 = self._pad_df_with_nans(df1, row_diff)
        elif row_diff > 0:
            df2 = self._pad_df_with_nans(df2, row_diff)

        offset, aligned_signal = self.align_signals(
            df1.loc[:, col1].fillna(0).values,
            df2.loc[:, col2].fillna(0).values
        )

        if offset > 0:
            nan_array = np.zeros((abs(offset), df1.shape[1]))
            nan_array[:] = np.nan
            df_nan = pd.DataFrame(nan_array)
            df_nan.columns = df1.columns
            df1 = pd.concat([df_nan, df1], axis=0)
        elif offset < 0:
            nan_array = np.zeros((abs(offset), df2.shape[1]))
            nan_array[:] = np.nan
            df_nan = pd.DataFrame(nan_array)
            df_nan.columns = df2.columns
            df2 = pd.concat([df_nan, df2], axis=0)

        duplicate_names = []
        for c in df1.columns:
            if c in df2.columns:
                duplicate_names.append(c)

        for duplicate_name in duplicate_names:
            df1 = df1.rename(columns={duplicate_name: duplicate_name + '_x'})
            df2 = df2.rename(columns={duplicate_name: duplicate_name + '_y'})

        df_aligned = pd.concat(
            [df1.reset_index(drop=True), df2.reset_index(drop=True)], 
            axis=1
        )

        if reference == 0:
            if col1 not in df_aligned.columns:
                col1 += '_x'
            df_aligned = df_aligned.dropna(subset=[col1])
        elif reference == 1:
            if col2 not in df_aligned.columns:
                col2 += '_y'
            df_aligned = df_aligned.dropna(subset=[col2])

        return df_aligned, offset
