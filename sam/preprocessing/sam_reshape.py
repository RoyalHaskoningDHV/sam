import pandas as pd


def sam_format_to_wide(data: pd.DataFrame, sep: str = "_"):
    """
    Converts a typical sam-format df '(TIME, ID, TYPE, VALUE)' to wide format.
    This is almost a wrapper around pd.pivot_table, although it does a few extra things.
    It removes the multiindex that would normally occur with ID + TYPE, by concatenating them
    with a separator in between.
    It also sorts the output by 'TIME', which is not guaranteed by pivot_table.

    Parameters
    ----------
    data: pd.DataFrame
        dataframe with TIME, ID, TYPE, VALUE columns
    sep: str, optional (default='_')
        separator that will be placed between ID and TYPE to create column names.

    Returns
    -------
    data_wide: pd.DataFrame
        the data, in wide format, with 1 column for every ID/TYPE combination, as well as a
        TIME column. For example, if ID is 'abc' and TYPEs are 'debiet' and 'stand', the
        created column names will be 'abc_debiet' and 'abc_stand', as well as TIME. The result
        will be sorted by TIME, ascending. The index will be a range from 0 to `nrows`.
    """

    data["ID"], data["TYPE"] = pd.Categorical(data["ID"]), pd.Categorical(data["TYPE"])

    data = data.pivot(values="VALUE", index=["TIME"], columns=["ID", "TYPE"])
    try:
        data.columns = [
            str(x[0]) + sep + x[1]
            for x in list(
                zip(
                    data.columns.levels[0][data.columns.codes[0]],
                    data.columns.levels[1][data.columns.codes[1]],
                )
            )
        ]
    except AttributeError:
        # In pandas 0.24, 'labels' was replaced by 'codes'
        # Since we support pandas 0.23, we need to fallback on labels if needed
        data.columns = [
            str(x[0]) + sep + x[1]
            for x in list(
                zip(
                    data.columns.levels[0][data.columns.labels[0]],
                    data.columns.levels[1][data.columns.labels[1]],
                )
            )
        ]
    data = data.reset_index().sort_values(by="TIME", axis=0)
    return data


def wide_to_sam_format(
    data: pd.DataFrame, sep: str = "_", idvalue: str = "", timecol: str = "TIME"
):
    """
    Convert a wide format dataframe to sam format.
    This function has the requirement that the dataframe has a time column, of which the name
    is given by `timecol`. Furthermore, the TYPE/ID combinations should be present in value
    column names that look like ID(sep)TYPE.

    If `sep` is `None`, then all column names (except `timecol) are assumed to be TYPE, with no
    id, so id will always be set to `idvalue`.

    Columns that look like 'A(sep)B(sep)C' will be split as 'ID = A, TYPE = B(sep)C'
    Columns that look like 'A' (without sep) will be split as 'ID = idvalue, TYPE = A'

    Parameters
    ----------
    data: pd.DataFrame
        data in wide format, with time column and other column names like 'ID(sep)TYPE'
    sep: string, optional (default='_')
        the seperator that appears in column names between id and type.
    idvalue: string, optional (default='')
        the default id value that is used when a column name contains no id
    timecol: string, optional (default='TIME')
        the column name of the time column that must be present

    Returns
    -------
    df: pd.DataFrame
        the data in sam format, with columns TIME, ID, TYPE, VALUE.
    """
    data = (
        data.rename({timecol: "TIME"}, axis=1)
        .set_index("TIME")
        .unstack()
        .reset_index()
        .rename({"level_0": "TYPE", 0: "VALUE"}, axis=1)
    )
    if sep is None:
        # The data only has one id
        data["ID"] = idvalue
        return data[["TIME", "ID", "TYPE", "VALUE"]]
    id_type = data["TYPE"].str.split(sep, n=1, expand=True)
    data["ID"] = id_type[0].replace("", idvalue)
    data["TYPE"] = id_type[1]

    # Ensure that A (without sep) will be split as ID = `idvalue`, TYPE = A
    missingtypes = data["TYPE"].isnull()
    data.loc[missingtypes, "TYPE"] = data["ID"].loc[missingtypes]
    data.loc[missingtypes, "ID"] = idvalue
    # Reorder columns
    return data[["TIME", "ID", "TYPE", "VALUE"]]
