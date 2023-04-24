import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random
from functools import lru_cache

def get_metacols(df):
    """return a list of metadata columns"""
    return [c for c in df.columns if c.startswith("Metadata_")]


def get_featurecols(df):
    """returna  list of featuredata columns"""
    return [c for c in df.columns if not c.startswith("Metadata")]


def get_metadata(df):
    """return dataframe of just metadata columns"""
    return df[get_metacols(df)]


def get_featuredata(df):
    """return dataframe of just featuredata columns"""
    return df[get_featurecols(df)]

def remove_negcon_and_empty_wells(df):
    """return dataframe of non-negative control wells"""
    df = (
        df.query('Metadata_control_type!="negcon"')
        .dropna(subset=["Metadata_broad_sample"])
        .reset_index(drop=True)
    )
    return df


def remove_empty_wells(df):
    """return dataframe of non-empty wells"""
    df = df.dropna(subset=["Metadata_broad_sample"]).reset_index(drop=True)
    return df


class AveragePrecision_non_vectorized(object):
    """
    Calculate average precision
    Parameters:
    -----------
    profile: pandas.DataFrame of profiles
    match_dict: dictionary with information about matching profiles
    reference_dict: dictionary with information about reference profiles
    n_reference: number of reference profiles
    random_baseline_ap: pandas.DataFrame with average precision of random baseline
    anti_match: boolean, if True, calculate anti-match average precision
    """

    def __init__(
        self,
        profile,
        match_dict,
        reference_dict,
        n_reference,
        random_baseline_ap,
        anti_match=False,
    ):
        self.profile = profile
        self.match_dict = match_dict
        self.reference_dict = reference_dict
        self.n_reference = n_reference
        self.random_baseline_ap = random_baseline_ap
        self.anti_match = anti_match

        self.ap = self.calculate_average_precision()
        self.map = self.calculate_mean_AP(self.ap)
        self.fp = self.calculate_fraction_positive(self.map)

    def calculate_average_precision(self):
        """
        Calculate average precision
        Returns:
        -------
        ap_df: dataframe with average precision values
        """
        _ap_df = pd.DataFrame(
            columns=self.match_dict["matching"]
            + ["n_matches", "n_reference", "ap", "correction", "ap_corrected"]
        )
        # Filter out profiles
        if "filter" in self.match_dict:
            profile_matching = self.filter_profiles(self.profile, self.match_dict)
        else:
            profile_matching = self.profile.copy()

        if "filter" in self.reference_dict:
            profile_reference = self.filter_profiles(self.profile, self.reference_dict)
        else:
            profile_reference = self.profile.copy()

        for group_index, group in tqdm(
            profile_matching.groupby(self.match_dict["matching"])
        ):
            for index, row in group.iterrows():
                _ap_dict = {}
                profile_matching_remaining = group.drop(index)

                # Remove matches that match columns of the query
                if "non_matching" in self.match_dict:
                    profile_matching_remaining = self.remove_non_matching_profiles(
                        row, profile_matching_remaining, self.match_dict
                    )

                # Keep those reference profiles that match columns of the query
                if "matching" in self.reference_dict:
                    query_string = " and ".join(
                        [f"{_}==@row['{_}']" for _ in self.reference_dict["matching"]]
                    )
                    if not query_string == "":
                        profile_reference_remaining = profile_reference.query(
                            query_string
                        ).reset_index(drop=True)
                else:
                    profile_reference_remaining = profile_reference.copy()

                # Remove those reference profiles that do not match columns of the query
                if "non_matching" in self.reference_dict:
                    profile_reference_remaining = self.remove_non_matching_profiles(
                        row, profile_reference_remaining, self.reference_dict
                    )

                # subsample reference
                k = min(self.n_reference, len(profile_reference_remaining))
                profile_reference_remaining = profile_reference_remaining.sample(
                    k
                ).reset_index(drop=True)

                # Combine dataframes
                profile_combined = pd.concat(
                    [profile_matching_remaining, profile_reference_remaining], axis=0
                ).reset_index(drop=True)

                # Extract features
                query_perturbation_features = row[
                    ~self.profile.columns.str.startswith("Metadata")
                ]
                profile_combined_features = get_featuredata(profile_combined)

                # Compute cosine similarity
                y_true = [1] * len(profile_matching_remaining) + [0] * len(
                    profile_reference_remaining
                )

                if np.sum(y_true) == 0:
                    continue
                else:
                    y_pred = cosine_similarity(
                        query_perturbation_features.values.reshape(1, -1),
                        profile_combined_features,
                    )[0]

                    if self.anti_match:
                        y_pred = np.abs(y_pred)

                    score = average_precision_score(y_true, y_pred)

                # Correct ap using the random baseline ap

                n_matches = np.sum(y_true)
                n_reference = k

                if (
                    self.random_baseline_ap.query(
                        "n_matches == '@n_matches' and n_reference == '@n_reference'"
                    ).empty
                    == True
                    and n_matches != 0
                ):
                    self.compute_random_baseline(n_matches, n_reference)

                    correction = self.random_baseline_ap.query(
                        "n_matches == @n_matches and n_reference == @n_reference"
                    )["ap"].quantile(0.95)

                else:
                    correction = 0

                for match in self.match_dict["matching"]:
                    _ap_dict[match] = row[match]
                _ap_dict["n_matches"] = int(n_matches)
                _ap_dict["n_reference"] = int(n_reference)
                _ap_dict["ap"] = score
                _ap_dict["correction"] = correction
                _ap_dict["ap_corrected"] = score - correction
                _ap_df = pd.concat(
                    [_ap_df, pd.DataFrame(_ap_dict, index=[0])],
                    axis=0,
                    ignore_index=True,
                )

        return _ap_df

    def compute_random_baseline(self, n_matches, n_reference):
        """
        Compute the random baseline for the average precision score
        Parameters
        ----------
        n_matches: int
            Number of matches
        n_reference: int
            Number of reference profiles
        """
        if (
            self.random_baseline_ap.query(
                "n_matches == @n_matches and n_reference == @n_reference"
            ).empty
            == True
        ):
            ranked_list = [i for i in range(n_matches + n_reference)]
            truth_values = [1 for i in range(n_matches)] + [
                0 for i in range(n_reference)
            ]

            for _ in range(10000):  # number of random permutations
                random.shuffle(ranked_list)
                random.shuffle(truth_values)

                self.random_baseline_ap = pd.concat(
                    [
                        self.random_baseline_ap,
                        pd.DataFrame(
                            {
                                "ap": average_precision_score(
                                    truth_values, ranked_list
                                ),
                                "n_matches": [n_matches],
                                "n_reference": [n_reference],
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )

    @staticmethod
    def filter_profiles(_profiles, _dict):
        """
        Filter profiles based on the filter dictionary
        Parameters
        ----------
        _profiles : pandas.DataFrame of profiles
        _dict : dictionary with filter columns
        Returns
        -------
        _profiles : pandas.DataFrame of filtered profiles
        """
        query_string = " and ".join(
            [
                " and ".join([f"{k}!={vi}" for vi in v])
                for k, v in _dict["filter"].items()
            ]
        )
        if not query_string == "":
            _profiles = _profiles.query(query_string).reset_index(drop=True)
        return _profiles

    @staticmethod
    def remove_non_matching_profiles(_query_profile, _profiles, _dict):
        """
        Remove profiles that match the query profile in the non_matching columns
        Parameters
        ----------
        _query_profile : pandas.Series of query profile
        _profiles : pandas.DataFrame of profiles
        _dict : dictionary with non_matching columns
        Returns
        -------
        _profiles : pandas.DataFrame of filtered profiles
        """
        for _ in _dict["non_matching"]:
            matching_col = [_query_profile[_] for i in range(len(_profiles))]
            _profiles = _profiles.loc[
                [
                    len(np.intersect1d(x[0].split("|"), x[1].split("|"))) == 0
                    for x in zip(_profiles[_], matching_col)
                ]
            ]
        return _profiles

    def calculate_mean_AP(self, _ap):
        """
        Calculate the mean average precision
        Parameters
        ----------
        _ap : pandas.DataFrame of average precision values
        Returns
        -------
        _map_df : pandas.DataFrame of mAP values gropued by matching columns
        """
        _map_df = (
            _ap.groupby(self.match_dict["matching"])
            .ap_corrected.mean()
            .reset_index()
            .rename(columns={"ap_corrected": "mAP"})
        )
        return _map_df

    @staticmethod
    def calculate_fraction_positive(_map_df):
        """
        Calculate the fraction of positive matches
        Parameters
        ----------
        _map_df : pandas.DataFrame of mAP values
        Returns
        -------
        _fp : float of fraction positive
        """
        _fp = len(_map_df.query("mAP>0")) / len(_map_df)
        return _fp

def time_point(modality, time_point):
    """
    Convert time point in hr to long or short time description
    Parameters:
    -----------
    modality: str
        perturbation modality
    time_point: int
        time point in hr
    Returns:
    -------
    str of time description
    """
    if modality == "compound":
        if time_point == 24:
            time = "short"
        else:
            time = "long"
    elif modality == "orf":
        if time_point == 48:
            time = "short"
        else:
            time = "long"
    else:
        if time_point == 96:
            time = "short"
        else:
            time = "long"

    return time

class AveragePrecision(object):
    """Compute average precision for a given query profile
    Parameters
    ----------
    profile : pandas.DataFrame
        A dataframe containing the query profile and reference profiles
    pos_dict : dict
        A dictionary containing the matching and non-matching columns for the query profile. The dictionary also contains a filter to remove profiles.
    ref_dict : dict
        A dictionary containing the matching and non-matching columns for the reference profiles. The dictionary also contains a filter to remove profiles.
    n_shuffle : int
        Number of times to shuffle to generate the random baseline.
    anti_match : bool
        If True, both matches and anti-matches are used to compute the average precision.
    multilabel : str
        If not empty, the "|" separated multilabel column is expanded.
    """

    def __init__(
        self,
        profile,
        pos_dict,
        ref_dict,
        n_shuffle=10000,
        anti_match=False,
        multilabel="",
    ):
        self.profile = profile
        self.pos_dict = pos_dict
        self.ref_dict = ref_dict
        self.n_shuffle = n_shuffle
        self.anti_match = anti_match
        self.multilabel = multilabel

        # Process dictionaries
        self.pos_dict = self.process_dictionaries(self.pos_dict)
        self.ref_dict = self.process_dictionaries(self.ref_dict)

        # Separate metadata and feature data

        self.meta_cols = get_metacols(self.profile)
        self.feature_cols = get_featurecols(self.profile)
        self.feature_data = get_featuredata(self.profile)

        # Compute metrics
        self.ap = self.compute_average_precision()

    @staticmethod
    def process_dictionaries(_dict):
        """Process dictionaries to ensure that all keys are present
        Parameters
        ----------
        _dict : dict
            Input dictionary for the average precision computation.
        Returns
        -------
        _dict : dict
            A dictionary containing the matching and non-matching columns for the query profile. The dictionary also contains a filter to remove profiles.
        """
        if not "filter" in _dict:
            _dict["filter"] = {}
        if not "matching_col" in _dict:
            _dict["matching_col"] = []
        if not "non_matching_col" in _dict:
            _dict["non_matching_col"] = []
        return _dict

    def compute_average_precision(self):
        """
        Compute average precision for a given query profile
        Returns
        -------
        average_precision_df : pandas.DataFrame
            A dataframe containing the average precision, among other columns, for the query profile.
        """
        # Filter profiles
        if not self.pos_dict["filter"] == {}:
            profile_query = (
                self.filter_profiles(self.profile, self.pos_dict)
                .reset_index()
                .rename(columns={"index": "query_index"})
                .assign(index=lambda x: x.query_index)
            )
        else:
            profile_query = self.profile.reset_index().rename(
                columns={"index": "query_index"}.assign(index=lambda x: x.query_index)
            )

        if not self.ref_dict["filter"] == {}:
            profile_ref = self.filter_profiles(
                self.profile, self.ref_dict
            ).reset_index()
        else:
            profile_ref = self.profile.reset_index()

        # Define matching and non-matching columns

        pos_matching_col = self.pos_dict["matching_col"]
        pos_non_matching_col = self.pos_dict["non_matching_col"]
        ref_matching_col = self.ref_dict["matching_col"]
        ref_non_matching_col = self.ref_dict["non_matching_col"]

        # Handle multi label columns

        if self.multilabel != "":
            profile_query = self.expanded_multilabel_columns(
                profile_query, self.multilabel
            )
            profile_query = self.expanded_multilabel_columns(
                profile_query, self.multilabel
            )
            profile_ref = self.expanded_multilabel_columns(profile_ref, self.multilabel)
            profile_ref = self.expanded_multilabel_columns(profile_ref, self.multilabel)

        # Create dataframes with metadata columns

        profile_query_meta = profile_query[
            ["query_index", "index"] + self.meta_cols
        ].copy()

        profile_ref_meta = profile_ref[["index"] + self.meta_cols].copy()

        # Identify matching and non-matching profiles

        pos_matches_non_matches = self.compute_matches_non_matches(
            profile_query_meta,
            profile_query_meta,
            self.meta_cols,
            pos_matching_col,
            pos_non_matching_col,
        )

        ref_matches_non_matches = self.compute_matches_non_matches(
            profile_query_meta,
            profile_ref_meta,
            self.meta_cols,
            ref_matching_col,
            ref_non_matching_col,
        )

        # Compute average precision

        average_precision_df = self.compute_ap(
            pos_matches_non_matches, ref_matches_non_matches
        )

        return average_precision_df

    @staticmethod
    def filter_profiles(_profiles, _dict):
        """
        Filter profiles based on a dictionary
        Parameters
        ----------
        _profiles : pandas.DataFrame
            A dataframe containing the profiles.
        _dict : dict
            A dictionary containing the filter to remove profiles.

        """
        query_string = " and ".join(
            [
                " and ".join([f"{k}!={vi}" for vi in v])
                for k, v in _dict["filter"].items()
            ]
        )
        if not query_string == "":
            _profiles = _profiles.query(query_string)
        return _profiles

    @staticmethod
    def expanded_multilabel_columns(profile, multilabel_column):
        """
        Expand multilabel columns
        Parameters
        ----------
        profile : pandas.DataFrame
            A dataframe containing the metadata.
        multilabel_column : str
            The name of the multilabel column.
        Returns
        -------
        profile : pandas.DataFrame
            A dataframe containing the metadata with the multilabel column expanded.
        """
        profile = (
            profile.assign(expanded=lambda x: x[multilabel_column].str.split("|"))
            .explode("expanded")
            .reset_index(drop=True)
            .drop(columns=multilabel_column)
            .rename(columns={"expanded": multilabel_column})
        )
        return profile

    @staticmethod
    def compute_matches_non_matches(
        df1, df2, meta_cols, matching_col, non_matching_col
    ):
        """
        Compute matches and non-matches for all query profiles
        Parameters
        ----------
        df1 : pandas.DataFrame
            A dataframe containing the query profiles.
        df2 : pandas.DataFrame
            A dataframe containing the query or reference profiles.
        meta_cols : list
            A list containing all the metadata columns.
        matching_col : list
            A list containing all the matching columns.
        non_matching_col : list
            A list containing all the non-matching columns.
        Returns
        -------
        df : pandas.DataFrame
            A dataframe containing the matches and non-matches for all query profiles.
        """
        # Identify matches
        if not matching_col == []:
            df = (
                df1[["query_index"] + matching_col]
                .merge(df2[["index"] + matching_col], on=matching_col, how="left")
                .drop(columns=matching_col)
                .merge(df1[["query_index"] + meta_cols], on="query_index", how="left")
            )
        else:
            df = (
                df1[["query_index"]]
                .assign(join_col=1)
                .merge(
                    df2[["index"]].assign(join_col=1),
                    on="join_col",
                    how="left",
                )
                .drop(columns="join_col")
                .merge(df1[["query_index"] + meta_cols], on="query_index", how="left")
            )

        # Remove self matches

        df = df.query("query_index!=index").reset_index(drop=True)

        # Identify non-matches

        if not non_matching_col == []:
            # Identify non-matches
            df_non_match = (
                df[["query_index", "index"] + non_matching_col]
                .merge(
                    df2[["index"] + non_matching_col],
                    on=["index"] + non_matching_col,
                    how="left",
                    indicator=True,
                )
                .query('_merge=="left_only"')
                .drop(columns=non_matching_col + ["_merge"])
            )

            # Identify non-matches that are matches so that they can be removed from the non-matches
            df_non_match_match = (
                df[["query_index", "index"] + non_matching_col]
                .merge(
                    df2[["index"] + non_matching_col],
                    on=["index"] + non_matching_col,
                    how="inner",
                )
                .drop(columns=non_matching_col)
            )

            df = (
                df_non_match.merge(
                    df_non_match_match,
                    on=["query_index", "index"],
                    how="left",
                    indicator=True,
                )
                .query('_merge=="left_only"')
                .drop(columns=["_merge"])
                .merge(df1[["query_index"] + meta_cols], on="query_index", how="left")
            )

        return df

    def compute_ap(self, pos_df, ref_df):
        """
        Compute average precision
        Parameters
        ----------
        pos_df : pandas.DataFrame
            A dataframe containing all the positive matches for all query profiles.
        ref_df : pandas.DataFrame
            A dataframe containing all the matches to the reference profiles for all query profiles.
        Returns
        -------
        df : pandas.DataFrame
            A dataframe containing the average precision, among other columns, for all query profiles.
        """
        df = (
            pos_df.groupby(["query_index"] + self.pos_dict["matching_col"])["index"]
            .apply(lambda x: np.unique(x).tolist())
            .reset_index()
            .assign(
                truth=lambda x: x["index"].apply(lambda y: [1 for _ in range(len(y))])
            )
            .assign(n_truth=lambda x: x["truth"].apply(lambda y: len(y)))
            .merge(
                ref_df.groupby(["query_index"] + self.pos_dict["matching_col"])["index"]
                .apply(lambda x: np.unique(x).tolist())
                .reset_index()
                .assign(
                    truth=lambda x: x["index"].apply(
                        lambda y: [0 for _ in range(len(y))]
                    )
                )
                .assign(n_truth=lambda x: x["truth"].apply(lambda y: len(y))),
                on=["query_index"] + self.pos_dict["matching_col"],
                how="left",
            )
            .dropna()  # Remove queries that do not have any positive or reference matches
            .reset_index(drop=True)
            .astype({"n_truth_x": "int64", "n_truth_y": "int64"})
            .assign(
                combined_index=lambda x: x.apply(
                    lambda y: y["index_x"] + y["index_y"],
                    axis=1,
                )
            )  # Combine positive and reference match indices
            .assign(
                combined_truth=lambda x: x.apply(
                    lambda y: y["truth_x"] + y["truth_y"],
                    axis=1,
                )
            )  # Combine positive and reference match truth values
            .assign(
                cosine_sim=lambda x: x.apply(
                    lambda y: cosine_similarity(
                        self.feature_data.values[[y["query_index"]]],
                        self.feature_data.values[y["combined_index"]],
                    )[0]
                    if not self.anti_match
                    else np.abs(
                        cosine_similarity(
                            self.feature_data.values[[y["query_index"]]],
                            self.feature_data.values[y["combined_index"]],
                        )[0]
                    ),
                    axis=1,
                )
            )  # Compute cosine similarity
            .assign(
                average_precision=lambda x: x.apply(
                    lambda y: average_precision_score(
                        y_true=y["combined_truth"], y_score=y["cosine_sim"]
                    ),
                    axis=1,
                )
            )  # Compute average precision
            .assign(
                random_baseline=lambda x: x.apply(
                    lambda y: self.compute_random_baseline(
                        y["n_truth_x"], y["n_truth_y"]
                    ),
                    axis=1,
                )
            )  # Compute random baseline
            .assign(
                pvalue=lambda x: x.apply(
                    lambda y: self.compute_p_value(
                        y["average_precision"],
                        y["random_baseline"],
                    ),
                    axis=1,
                )
            )  # Compute p-value
            .assign(
                neglog10pvalue=lambda x: x.apply(
                    lambda y: -np.log10(y["pvalue"]),
                    axis=1,
                )
            )  # Compute negative log10 p-value
            .assign(
                random_baseline_adjustment=lambda x: x.apply(
                    lambda y: self.compute_random_baseline_adjustment(
                        y["random_baseline"],
                    ),
                    axis=1,
                )
            )  # Compute random baseline adjustment
            .assign(
                adjusted_average_precision=lambda x: x.apply(
                    lambda y: (
                        y["average_precision"] - y["random_baseline_adjustment"]
                    ),
                    axis=1,
                ),
            )  # Compute adjusted average precision
            .rename(
                columns={
                    "index_x": "pos_index",
                    "index_y": "ref_index",
                    "truth_x": "pos_truth",
                    "truth_y": "ref_truth",
                    "n_truth_x": "n_pos_truth",
                    "n_truth_y": "n_ref_truth",
                }
            )  # Rename columns
        )

        return df

    @lru_cache(maxsize=None)
    def compute_random_baseline(self, n_pos, n_reference):
        """
        Compute random baseline
        Parameters
        ----------
        n_pos : int
            Number of positive matches.
        n_reference : int
            Number of reference matches.
        Returns
        -------
        ap : list
            A list of average precision values for each random permutation.
        """
        ranked_list = [i for i in range(n_pos + n_reference)]
        truth_values = [1 for i in range(n_pos)] + [0 for i in range(n_reference)]

        ap = []

        for _ in range(self.n_shuffle):  # number of random permutations
            random.shuffle(ranked_list)
            random.shuffle(truth_values)
            ap.append(average_precision_score(truth_values, ranked_list))

        return ap

    @staticmethod
    def compute_p_value(ap, random_baseline_ap):
        """
        Compute p-value
        Parameters
        ----------
        ap : float
            Average precision.
        random_baseline_ap : list
            A list of average precision values for each random permutation.
        Returns
        -------
        pvalue : float
            The p-value.
        """
        return (1 + np.sum(np.asarray(random_baseline_ap) > ap)) / (
            1 + len(random_baseline_ap)
        )

    @staticmethod
    def compute_random_baseline_adjustment(random_baseline_ap):
        """
        Compute random baseline adjustment
        Parameters
        ----------
        random_baseline_ap : list
            A list of average precision values for each random permutation.
        Returns
        -------
        random_baseline_adjustment : float
            The random baseline adjustment.
        """
        return np.quantile(random_baseline_ap, 0.95)