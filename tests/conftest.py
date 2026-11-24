import warnings

import pytest

from faircredit.data import FEATURES, TARGET, load_german_credit

warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture(scope="session")
def df():
    return load_german_credit()


@pytest.fixture(scope="session")
def Xy(df):
    return df[FEATURES], df[TARGET]
