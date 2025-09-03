import numpy as np
import pandas as pd
import polars as pl
import pytest

from lightshap._utils import get_dataclass, get_polars


class TestGetDataclass:
    """Test the get_dataclass utility function."""

    def test_numpy_array(self):
        """Test that numpy arrays are correctly identified."""
        data = np.array([[1, 2], [3, 4]])
        result = get_dataclass(data)
        assert result == "np"

    def test_pandas_dataframe(self):
        """Test that pandas DataFrames are correctly identified."""
        data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = get_dataclass(data)
        assert result == "pd"

    def test_polars_dataframe(self):
        """Test that polars DataFrames are correctly identified."""
        data = pl.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = get_dataclass(data)
        assert result == "pl"

    def test_unknown_type_error(self):
        """Test that unknown data types raise KeyError."""
        data = {"A": [1, 2], "B": [3, 4]}  # Plain dict
        with pytest.raises(KeyError, match="Unknown data class"):
            get_dataclass(data)


def test_get_polars_success():
    """Test that get_polars returns polars module when available."""
    result = get_polars()
    assert result is pl
