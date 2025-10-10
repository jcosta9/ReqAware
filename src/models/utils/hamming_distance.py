"""
Helper function to calculate the Hamming distance

Copyright (C) 2024  Joao Paulo Costa de Araujo

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; If not, see <http://www.gnu.org/licenses/>.
"""

from typing import List
import numpy as np
import pandas as pd
import scipy.spatial.distance as distance


def generate_concept_label_df(
    labels: list, concepts: List[List[int]], classes_names: dict, concepts_names: list
) -> pd.DataFrame:
    """Creates a dataframe that contain both label and concepts

    Args:
        labels (list): list of labels. Each item is a label associated with the that particular index
        concepts (List[List[int]]): list of concepts. Each sublist is the set of concepts associated with the
                                        that particular index.
        classes_names (dict): Dictionary for mapping class ids with class names
        concepts_names (list): ordered list of concepts

    Returns:
        pd.DataFrame: Table that associates samples (index) with labels and concepts.
    """

    y = pd.DataFrame()
    y["class_id"] = labels
    y["class_name"] = y.class_id.map(classes_names)
    return y.join(pd.DataFrame(concepts, columns=concepts_names))


def generate_df_gt_predictions(
    y_true_concept: List[List[int]],
    y_pred_concept: List[List[int]],
    y_true_label: list,
    y_pred_label: list,
    classes_names: dict,
    concepts_names: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates label and concepts dataframe for ground truth and predictions

    Args:
    y_true_concept (List[List[int]]): list of ground truth concepts. Each sublist is the set of concepts associated
                                    with the that particular index.
    y_pred_concept (List[List[int]]): list of predictec concepts. Each sublist is the set of concepts associated
                                    with the that particular index.
    y_true_label (list): list of labels. Each item is a grount truth  label associated with the that particular index
    y_pred_label (list): list of labels. Each item is a predicted label associated with the that particular index
    classes_names (dict): Dictionary for mapping class ids with class names
    concepts_names (list): ordered list of concepts

    Returns:
    tuple[pd.DataFrame, pd.DataFrame]: Table that associates samples (index) with labels and concepts for both
                                    ground truth and predicted samples.
    """

    y_true = generate_concept_label_df(
        y_true_label, y_true_concept, classes_names, concepts_names
    )
    y_pred = generate_concept_label_df(
        y_pred_label, y_pred_concept, classes_names, concepts_names
    )
    return y_true, y_pred


def calculate_hamming(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, n_concepts=1
) -> pd.DataFrame:
    return (
        pd.DataFrame(
            [
                y_true.apply(lambda row: row.tolist(), axis=1),  # df to series of lists
                y_pred.apply(lambda row: row.tolist(), axis=1),
            ],
            index=["true", "pred"],
        ).T
    ).apply(lambda x: distance.hamming(x.true, x.pred) * n_concepts, axis=1)


def calculate_hamming_df(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, n_concepts=1
) -> pd.DataFrame:
    """Compute hamming distance by comparing the ground truth and predicted lists of concepts.

    Args:
        y_true (pd.DataFrame): Ground truth Concept-label dataframe
        y_pred (pd.DataFrame): Predicted concept-label dataframe
        n_concepts (int, optional): Number of concepts. Defaults to 1 for normalized distance

    Returns:
        pd.DataFrame: DataFrame with the following columns
            hamm: hamming distance between sets of ground truth and predicted concepts
            concept_match: set to 1 if the entire set of predicted concepts match with ground truth.
                            0 otherwise.
            label_match: set to 1 if a label matches with ground truth.
                            0 otherwise.

    """
    hamm_df = pd.DataFrame()

    hamm_df["hamm"] = calculate_hamming(
        y_true.iloc[:, 2:], y_pred.iloc[:, 2:], n_concepts
    )
    hamm_df["concept_match"] = hamm_df.hamm.map(lambda x: 1 if x == 0 else 0)
    hamm_df["label_match"] = (y_true.class_id == y_pred.class_id).astype(int)

    return hamm_df


def get_concept_label_alignment_matrix(hamm_df):
    a = hamm_df.query("concept_match==1 & label_match==1").shape[0]
    b = hamm_df.query("concept_match==1 & label_match==0").shape[0]
    c = hamm_df.query("concept_match==0 & label_match==1").shape[0]
    d = hamm_df.query("concept_match==0 & label_match==0").shape[0]

    return np.array([[a, c], [b, d]])
