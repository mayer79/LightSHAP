import matplotlib as mpl

mpl.use("Agg")  # Use non-interactive backend for testing

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from lightshap.explanation.explanation import Explanation
from lightshap.explanation.explanationplotter import ExplanationPlotter


def create_explanation(n_samples=10, n_features=5, n_outputs=1):
    """Create an Explanation object for testing."""
    rng = np.random.default_rng(0)
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Mixed-type features
    X_data = {}
    for i, name in enumerate(feature_names):
        if i == 0:  # First feature is categorical
            categories = ["A", "B", "C", "D"]
            values = rng.choice(categories, size=n_samples)
            X_data[name] = pd.Categorical(values)
        else:  # Other features are numeric
            X_data[name] = rng.random(n_samples)

    X = pd.DataFrame(X_data)

    # Add 10% missing values randomly across all features
    n_missing = int(0.1 * n_samples * n_features)
    missing_indices = rng.choice(n_samples * n_features, size=n_missing, replace=False)

    for idx in missing_indices:
        row_idx = idx // n_features
        col_idx = idx % n_features
        X.iloc[row_idx, col_idx] = np.nan

    if n_outputs == 1:
        shap_values = rng.random((n_samples, n_features))
        baseline = 0.5
    else:
        shap_values = rng.random((n_samples, n_features, n_outputs))
        baseline = rng.random(n_outputs)

    return Explanation(
        shap_values=shap_values, X=X, baseline=baseline, feature_names=feature_names
    )


class TestBar:
    """Test suite for the bar method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.explanation = create_explanation(n_samples=20, n_features=8)
        self.plotter = ExplanationPlotter(self.explanation)

    def test_returns_matplotlib_axis(self):
        """Test that bar returns a matplotlib Axes object."""
        ax = self.plotter.bar()
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_max_display_limits_features(self):
        """Test that max_display parameter limits the number of displayed features."""
        max_display = 3
        ax = self.plotter.bar(max_display=max_display)

        # Check that the number of y-tick labels is at most max_display
        y_labels = ax.get_yticklabels()
        assert len(y_labels) == max_display
        plt.close()

    def test_max_display_none_shows_all_features(self):
        """Test that max_display=None shows all features."""
        ax = self.plotter.bar(max_display=None)

        y_labels = ax.get_yticklabels()
        assert len(y_labels) == self.explanation.shape[1]  # n_features
        plt.close()

    def test_custom_axis(self):
        """Test that custom axis can be provided."""
        fig, ax = plt.subplots()
        fig.canvas.draw()  # Force canvas initialization
        result_ax = self.plotter.bar(ax=ax)

        assert result_ax is ax
        plt.close()

    def test_multi_output_explanation(self):
        """Test bar with multi-output explanation."""
        M = 3
        multi_explanation = create_explanation(n_samples=10, n_features=5, n_outputs=M)
        plotter = ExplanationPlotter(multi_explanation)

        ax = plotter.bar()
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_bar_containers_exist(self):
        """Test that bar containers are created."""
        ax = self.plotter.bar()

        # Check that there are bar containers
        assert len(ax.containers) > 0
        plt.close()

    def test_axis_labels_and_grid(self):
        """Test that proper axis labels and grid are set."""
        ax = self.plotter.bar()

        assert ax.get_xlabel() == "Mean Absolute SHAP Value"
        assert ax.grid  # Grid should be enabled
        plt.close()

    def test_no_bar_labels_with_zero_fontsize(self):
        """Test that no bar labels are added when label_fontsize=0."""
        ax = self.plotter.bar(label_fontsize=0.0)

        # Should still have bars but no labels
        assert len(ax.containers) > 0
        plt.close()

    def test_inverted_yaxis(self):
        """Test that y-axis is properly inverted for feature importance ranking."""
        ax = self.plotter.bar()

        # Y-axis should be inverted (higher importance features at top)
        ylim = ax.get_ylim()
        assert ylim[0] > ylim[1]  # Inverted axis
        plt.close()

    def test_invalid_ax_type_raises_error(self):
        """Test that invalid ax parameter raises TypeError."""
        with pytest.raises(TypeError, match="ax must be a matplotlib Axes"):
            self.plotter.bar(ax="invalid")

    def test_custom_color(self):
        """Test that custom color can be set."""
        custom_color = "#ff0000"
        ax = self.plotter.bar(color=custom_color)

        # Check that bars exist (actual color testing is complex with matplotlib)
        assert len(ax.containers) > 0
        plt.close()


class TestBeeswarm:
    """Test suite for the beeswarm method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.explanation = create_explanation(n_samples=20, n_features=8)
        self.plotter = ExplanationPlotter(self.explanation)

    def test_returns_matplotlib_axis(self):
        """Test that beeswarm returns a matplotlib Axes object."""
        ax = self.plotter.beeswarm()
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_max_display_limits_features(self):
        """Test that max_display parameter limits the number of displayed features."""
        max_display = 3
        ax = self.plotter.beeswarm(max_display=max_display)

        # Check that the number of y-tick labels is at most max_display
        y_labels = ax.get_yticklabels()
        assert len(y_labels) == max_display
        plt.close()

    def test_max_display_none_shows_all_features(self):
        """Test that max_display=None shows all features."""
        ax = self.plotter.beeswarm(max_display=None)

        y_labels = ax.get_yticklabels()
        assert len(y_labels) == self.explanation.shape[1]  # n_features
        plt.close()

    def test_custom_axis(self):
        """Test that custom axis can be provided."""
        fig, ax = plt.subplots()
        fig.canvas.draw()  # Force canvas initialization
        result_ax = self.plotter.beeswarm(ax=ax)

        assert result_ax is ax
        plt.close()

    def test_inverted_yaxis(self):
        """Test that y-axis is properly inverted for feature importance ranking."""
        ax = self.plotter.beeswarm()

        # Y-axis should be inverted (higher importance features at top)
        ylim = ax.get_ylim()
        assert ylim[0] > ylim[1]  # Inverted axis
        plt.close()

    def test_multi_output_explanation(self):
        """Test beeswarm with multi-output explanation."""
        multi_explanation = create_explanation(n_samples=10, n_features=5, n_outputs=3)
        plotter = ExplanationPlotter(multi_explanation)

        ax = plotter.beeswarm(which_output=1)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_colorbar_exists(self):
        """Test that a colorbar is created in the plot."""
        ax = self.plotter.beeswarm()
        fig = ax.get_figure()

        # Check that colorbar was added to the figure
        assert len(fig.axes) == 2  # Main axis + colorbar axis
        plt.close()

    def test_jitter_width_zero(self):
        """Test that jitter_width=0 works without errors."""
        ax = self.plotter.beeswarm(jitter_width=0)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_scatter_points_exist(self):
        """Test that scatter points are actually plotted."""
        ax = self.plotter.beeswarm()

        # Check that there are scatter plot collections
        collections = [c for c in ax.collections if hasattr(c, "get_offsets")]
        assert len(collections) > 0
        plt.close()

    def test_axis_labels_and_grid(self):
        """Test that proper axis labels and grid are set."""
        ax = self.plotter.beeswarm()

        assert ax.get_xlabel() == "SHAP Value"
        assert ax.grid  # Grid should be enabled
        plt.close()

    def test_invalid_ax_type_raises_error(self):
        """Test that invalid ax parameter raises TypeError."""
        with pytest.raises(TypeError, match="ax must be a matplotlib Axes"):
            self.plotter.beeswarm(ax="invalid")


class TestWaterfall:
    """Test suite for the waterfall method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.explanation = create_explanation(n_samples=20, n_features=8)
        self.plotter = ExplanationPlotter(self.explanation)

    def test_returns_matplotlib_axis(self):
        """Test that waterfall returns a matplotlib Axes object."""
        ax = self.plotter.waterfall(row_id=0)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_max_display_limits_features(self):
        """Test that max_display parameter limits the number of displayed features."""
        max_display = 2
        ax = self.plotter.waterfall(row_id=0, max_display=max_display)

        # Check that the number of y-tick labels is at most max_display
        y_labels = ax.get_yticklabels()
        assert len(y_labels) == max_display
        plt.close()

    def test_max_display_none_shows_all_features(self):
        """Test that max_display=None shows all features."""
        ax = self.plotter.waterfall(row_id=0, max_display=None)

        y_labels = ax.get_yticklabels()
        assert len(y_labels) == self.explanation.shape[1]  # n_features
        plt.close()

    def test_row_id_validation(self):
        """Test that invalid row_id raises ValueError."""
        with pytest.raises(ValueError, match="row_id .* must be integer"):
            self.plotter.waterfall(row_id=len(self.explanation))

        with pytest.raises(ValueError, match="row_id .* must be integer"):
            self.plotter.waterfall(row_id=-1)

    def test_fill_colors_validation(self):
        """Test that invalid fill_colors raises ValueError."""
        with pytest.raises(
            ValueError, match="fill_colors must be a list or tuple of length 2"
        ):
            self.plotter.waterfall(row_id=0, fill_colors=["red"])

        with pytest.raises(
            ValueError, match="fill_colors must be a list or tuple of length 2"
        ):
            self.plotter.waterfall(row_id=0, fill_colors="red")

    def test_annotation_validation(self):
        """Test that invalid annotation parameter raises ValueError."""
        with pytest.raises(
            ValueError, match="annotation must be a list or tuple of length 2"
        ):
            self.plotter.waterfall(row_id=0, annotation=["E[f(x)]"])

    def test_custom_axis(self):
        """Test that custom axis can be provided."""
        fig, ax = plt.subplots()
        fig.canvas.draw()  # Force canvas initialization
        result_ax = self.plotter.waterfall(row_id=0, ax=ax)

        assert result_ax is ax
        plt.close()

    def test_multi_output_explanation(self):
        """Test waterfall with multi-output explanation."""
        multi_explanation = create_explanation(n_samples=10, n_features=5, n_outputs=3)
        plotter = ExplanationPlotter(multi_explanation)

        ax = plotter.waterfall(which_output=1)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_max_display_creates_other_features_label(self):
        """Test that max_display creates 'other features' label when needed."""
        max_display = 3
        ax = self.plotter.waterfall(row_id=0, max_display=max_display)

        y_labels = [label.get_text() for label in ax.get_yticklabels()]

        # Should have max_display labels
        assert len(y_labels) == max_display

        # Last label should mention "other features" if we collapsed features
        if self.explanation.shape[1] > max_display:
            assert (
                f"{self.explanation.shape[1] - max_display + 1} other features"
                in y_labels[-1]
            )

        plt.close()


class TestScatter:
    """Test suite for the scatter method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.explanation = create_explanation(n_samples=20, n_features=8)
        self.plotter = ExplanationPlotter(self.explanation)

    def test_returns_matplotlib_axis(self):
        """Test that scatter returns a matplotlib Axes object."""
        ax = self.plotter.scatter(features=["feature_1", "feature_2"])
        assert isinstance(ax, np.ndarray)  # Returns array of axes for subplots
        assert isinstance(ax.flatten()[0], plt.Axes)
        assert len(ax.flatten()) == 2
        plt.close()

    def test_single_feature(self):
        """Test scatter plot with a single feature."""
        ax = self.plotter.scatter(features=["feature_1"])
        assert isinstance(ax, np.ndarray)
        assert len(ax.flatten()) == 1
        plt.close()

    def test_custom_axis_single(self):
        """Test that custom single axis can be provided."""
        fig, ax = plt.subplots()
        result_ax = self.plotter.scatter(features=["feature_1"], ax=ax)

        assert result_ax is ax
        plt.close()

    def test_custom_axis_array(self):
        """Test that custom axis array can be provided."""
        fig, axes = plt.subplots(1, 2)
        result_ax = self.plotter.scatter(features=["feature_1", "feature_2"], ax=axes)

        assert np.array_equal(result_ax, axes)
        plt.close()

    def test_multi_output_explanation(self):
        """Test scatter with multi-output explanation."""
        multi_explanation = create_explanation(n_samples=10, n_features=5, n_outputs=3)
        plotter = ExplanationPlotter(multi_explanation)

        ax = plotter.scatter(features=["feature_1"], which_output=1)
        assert isinstance(ax, plt.Axes | np.ndarray)
        plt.close()

    def test_color_features_empty_list(self):
        """Test scatter with color_features=[] uses the specified color."""
        custom_color = "#ff5733"
        ax = self.plotter.scatter(
            features=["feature_1"], color_features=[], color=custom_color
        )

        first_ax = ax.flatten()[0] if isinstance(ax, np.ndarray) else ax

        scatter_collections = [
            c for c in first_ax.collections if hasattr(c, "get_facecolors")
        ]
        assert len(scatter_collections) > 0

        # Get the colors of the scatter points
        face_colors = scatter_collections[0].get_facecolors()
        assert len(face_colors) > 0

        # Convert custom_color to RGBA for comparison
        expected_rgba = mcolors.to_rgba(custom_color)

        # Check if all points have the expected color (within tolerance)
        # Note: matplotlib sometimes adds alpha channel, so we compare RGB components
        for color in face_colors:
            assert np.allclose(color[:3], expected_rgba[:3], atol=0.01)

        plt.close()

    def test_axis_labels(self):
        """Test that proper axis labels are set."""
        ax = self.plotter.scatter(features=["feature_1"])

        # Get the first (or only) subplot
        first_ax = ax.flatten()[0] if isinstance(ax, np.ndarray) else ax
        assert first_ax.get_xlabel() == "feature_1"
        plt.close()

    def test_no_shared_y_axis(self):
        """Test sharey=False parameter functionality."""
        ax = self.plotter.scatter(features=["feature_1", "feature_2"], sharey=False)
        assert isinstance(ax, np.ndarray)
        plt.close()

    def test_invalid_ax_type_raises_error(self):
        """Test that invalid ax parameter raises TypeError."""
        with pytest.raises(
            TypeError, match="ax must be a matplotlib Axes or an array of Axes"
        ):
            self.plotter.scatter(features=["feature_1"], ax="invalid")

    def test_mismatched_axes_count_raises_error(self):
        """Test that mismatched number of axes and features raises ValueError."""
        # Create 2 axes but try to plot 3 features
        fig, axes = plt.subplots(1, 2)

        with pytest.raises(ValueError, match="Expected 3 axes, got 2"):
            self.plotter.scatter(
                features=["feature_1", "feature_2", "feature_3"],
                ax=axes,
            )

        plt.close()
