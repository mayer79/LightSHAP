import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

from lightshap.explanation._utils import (
    _check_features,
    beeswarm_jitter,
    color_axis_info,
    get_text_bbox,
    halton_sequence,
    min_max_scale,
    plot_layout,
    safe_to_float,
)


class ExplanationPlotter:
    """Plotting methods for Explanation objects."""

    def __init__(self, explanation):
        self._explanation = explanation

    def bar(
        self,
        max_display=50,
        color="#b40426",
        width=0.7,
        label_fontsize=10.0,
        ax=None,
        which_output=None,
        **kwargs,
    ):
        """
        Barplot of mean absolute SHAP values per feature.
        Accepts multi-output SHAP values.

        Parameters
        ----------
        max_display : int, optional
            Maximum number of features to include in the plot. Default is 50.
            Set to None to show all features.
        color : str, optional
            Color of the bars. Ignored if multiple outputs are shown.
            Default is "#b40426".
        width : float, optional
            Width of the bars. Default is 0.7.
        label_fontsize : float, optional
            Font size for the labels. Default is 10.0. Set to 0.0 for no labels.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axis to plot on. If None, a new figure is created.
        which_output : int or string, optional
            Index or name of the output dimension to plot.
            If None, all outputs are plotted. Only relevant for multi-output models.
        **kwargs : dict, optional
            Additional keyword arguments passed to barh().

        Returns
        -------
        matplotlib.axes.Axes
            Matplotlib axis with the feature importances.

        Example
        -------
        >>> import matplotlib.pyplot as plt
        >>> explanation = Explanation(...)
        >>> explanation.plot.bar(label_fontsize=0)
        >>> plt.savefig("filename.png", bbox_inches="tight")
        """
        imp = self._explanation.importance(which_output=which_output)

        if max_display is not None and max_display < imp.shape[0]:
            imp = imp[:max_display]

        # Figure shape
        if ax is None:
            height = 2 + imp.shape[0] * 0.5
            _, ax = plt.subplots(figsize=(6, height))
        elif not isinstance(ax, plt.Axes):
            msg = "ax must be a matplotlib Axes."
            raise TypeError(msg)

        # Color is used only with single-output plots
        if imp.ndim > 1:
            color = None

        barh = imp.plot.barh(ax=ax, color=color, width=width, zorder=2, **kwargs)

        if label_fontsize > 0:
            for container in barh.containers:
                ax.bar_label(container, padding=4, fmt="%.3g", fontsize=label_fontsize)
            _, right = ax.get_xlim()
            max_text_right = get_text_bbox(ax)[1]  # right
            ax.set_xlim(right=max(right, max_text_right) + 0.03 * right)

        ax.invert_yaxis()
        ax.set_xlabel("Mean Absolute SHAP Value")
        ax.grid(alpha=0.3)

        return ax

    def beeswarm(
        self,
        max_display=10,
        jitter_width=0.4,
        cmap="coolwarm",
        s=16.0,
        ax=None,
        which_output=None,
        **kwargs,
    ):
        """
        Beeswarm summary plot of SHAP values

        Colors represent feature values on a common numeric scale. Categorical
        features are converted to numeric values using the order of its categories.

        Parameters
        ----------
        max_display : int, optional
            Maximum number of features to include in the plot. Default is 10.
            Set to None to show all features.
        jitter_width : float, optional
            Width scaling factor for the jittering. Default is 0.4.
        cmap : str or matplotlib colormap, optional
            Colormap to use for coloring points. Default is 'coolwarm'.
        s : float, optional
            Size of the points in the scatter plot. Default is 16.0.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axis to use. If None, a new figure is created.
        which_output : int or string, optional
            Index or name of the output dimension to plot.
            If None, the last output is plotted. Only relevant for multi-output models.
        **kwargs : dict
            Additional keyword arguments passed to scatter().

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes containing the plot.

        Example
        -------
        >>> import matplotlib.pyplot as plt
        >>> explanation = Explanation(...)
        >>> explanation.plot.beeswarm()
        >>> plt.savefig("filename.png", bbox_inches="tight")

        """
        xp = self._explanation

        # Pick output dimension
        if xp.ndim == 3:
            if which_output is None:
                which_output = xp.shape[2] - 1  # pick last
                print("Selected last output. Use which_output to use a different one.")
            xp = xp.select_output(which_output)

        # Revert order for plotting and select most important features
        imp = xp.importance()
        if max_display is not None and max_display < imp.shape[0]:
            imp = imp[:max_display]

        # Turn feature values numeric and to identical scale from 0 to 1
        features = imp.index.to_list()
        X_scaled = xp.X[features].apply(safe_to_float).apply(min_max_scale)
        p = len(features)

        # Used for neat jitter - same for all features
        halton_vals = halton_sequence(xp.shape[0])

        # ===========================================================================
        # Plot
        # ===========================================================================

        if ax is None:
            height = 2 + p * 0.5
            fig, ax = plt.subplots(figsize=(7, height))
        elif not isinstance(ax, plt.Axes):
            msg = "ax must be a matplotlib Axes."
            raise TypeError(msg)
        else:
            fig = ax.get_figure()

        cmap = plt.get_cmap(cmap)
        cmap.set_bad("gray", alpha=kwargs.get("alpha", 1.0))  # Missings are gray

        # Draw each univariate scatter plot separately
        for i, feature in enumerate(features):
            idx = xp.feature_names.index(feature)
            xi = xp.shap_values[:, idx]
            yi = np.full_like(xi, fill_value=i)
            if jitter_width > 0:
                yi += jitter_width * beeswarm_jitter(xi, halton_vals=halton_vals)

            scatter = ax.scatter(
                xi,
                yi,
                c=X_scaled[feature],
                s=s,
                cmap=cmap,
                vmin=0,
                vmax=1,
                plotnonfinite=True,
                zorder=3,
                **kwargs,
            )

        # Color bar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
        cbar.ax.set_yticks(ticks=[0, 1], labels=["Low", "High"])
        cbar.set_label("Feature value", labelpad=-16.5)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(size=0)

        # Layout
        ax.set_yticks(range(p), labels=features)
        ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
        ax.set_xlabel("SHAP Value")
        ax.invert_yaxis()
        ax.grid(alpha=0.3, zorder=0)

        return ax

    def scatter(
        self,
        features=None,
        color_features=None,
        color="#9ca5df",
        s=16.0,
        jitter_width=0.3,
        rotate_xlabels=None,
        cmap="coolwarm",
        max_color_labels=10,
        max_color_label_length=15,
        max_xlabels=20,
        max_xlabel_length=15,
        sharey=True,
        ax=None,
        which_output=None,
        **kwargs,
    ):
        """
        SHAP dependence plots

        For each feature, its SHAP values are plotted against the feature values.
        If `color_features` is not `[]`, color features are selected according to
        a correlation based interaction strength heuristic.

        Parameters
        ----------
        features : list-like, optional
            List-like of feature names to plot. By default, all features are used.
        color_features : list-like, optional
            List-like of feature names to use for coloring the points.
            For each subplot, a color feature will be selected from `color_features`
            using a heuristic that measures potential interaction strength.
            By default, all features are used.
            Set `color_features=[]` to not use any feature for coloring.
        color : str, optional
            Color of the points in the scatter plot. Used if `color_features` is [] or
            if all interaction heuristics are zero for the given feature.
            Default is "#9ca5df".
        s : float, optional
            Size of the points in the scatter plot. Default is 16.0.
        jitter_width : float, optional
            Width of the jitter for discrete features. Default is 0.3.
        rotate_xlabels : float, optional
            Rotation angle for non-numeric x-axis labels. Default is None, which means
            no rotation for up to four labels, and 45 or 90 degrees for more.
        cmap : str, optional
            Colormap to use for coloring the points. Default is "coolwarm".
        max_color_labels : int, optional
            Maximum number of labels to show on the color bar. Default is 10.
            Only relevant for categorical features.
        max_color_label_length : int, optional
            Maximum length of categorical labels on the color bar. Default is 15.
            Only relevant for categorical features.
        max_xlabels : int, optional
            Maximum number of labels to show on the x-axis. Default is 20.
            Only relevant for categorical features.
        max_xlabel_length : int, optional
            Maximum length of categorical labels on the x-axis. Default is 15.
            Only relevant for categorical features.
        sharey : bool, optional
            Whether to share the y-axis across subplots. Default is True.
        ax : matplotlib.axes.Axes or array of matplotlib.axes.Axes, optional
            Matplotlib axis or array of axes to plot on.
            If None, a new figure is created.
        which_output : int or string, optional
            Index or name of the output dimension to plot.
            If None, the last output is plotted. Only relevant for multi-output models.
        **kwargs : dict, optional
            Additional keyword arguments passed to scatter().

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray
            Matplotlib axis or array of axes with the dependence plots.

        Example
        -------
        >>> import matplotlib.pyplot as plt
        >>> explanation = Explanation(...)
        >>> explanation.plot.scatter(sharey=False)
        >>> plt.savefig("filename.png", bbox_inches="tight")
        """
        xp = self._explanation

        # Pick output dimension
        if xp.ndim == 3:
            if which_output is None:
                which_output = xp.shape[2] - 1  # pick last
                print("Selected last output. Use which_output to use a different one.")
            xp = xp.select_output(which_output)

        # Prepare feature list and color feature list
        features = _check_features(features, xp.feature_names, name="features")
        color_features = _check_features(
            color_features, xp.feature_names, name="color features"
        )

        # For each feature, we need the corresponding color feature (or None)
        color_feature_dict = {}
        if len(color_features) > 1:
            H = xp.interaction_heuristic(features, color_features=color_features)
            for feature in features:
                H_feature = H.loc[feature, :]
                if H_feature.max() > 0:
                    color_feature_dict[feature] = H_feature.idxmax()
        elif len(color_features) == 1:
            for feature in features:
                if feature != color_features[0]:
                    color_feature_dict[feature] = color_features[0]

        # Color axis information for each color feature in color_feature_dict
        color_axis_dict = {}
        for color_feature in set(color_feature_dict.values()):
            color_axis_dict[color_feature] = color_axis_info(
                xp.X[color_feature],
                cmap=cmap,
                max_color_labels=max_color_labels,
                max_color_label_length=max_color_label_length,
                **kwargs,
            )

        # Set up plot
        cleanup = False
        single_axis_as_array = False

        if ax is None:
            nrows, ncols = plot_layout(len(features))
            width = 6.4 * np.sqrt(ncols)
            height = 4.8 / 6.4 * nrows * width / ncols

            cleanup = nrows * ncols > len(features)

            fig, ax = plt.subplots(
                ncols=ncols,
                nrows=nrows,
                figsize=(width, height),
                layout="compressed",
                sharey=sharey,
                squeeze=False,
            )
        else:
            if isinstance(ax, plt.Axes):
                fig = ax.get_figure()
                single_axis_as_array = True
                ax = np.array([ax])
            elif isinstance(ax, np.ndarray) and isinstance(ax.flatten()[0], plt.Axes):
                fig = ax.flatten()[0].get_figure()
            else:
                msg = "ax must be a matplotlib Axes or an array of Axes."
                raise TypeError(msg)
            if len(ax.flatten()) != len(features):
                msg = f"Expected {len(features)} axes, got {len(ax.flatten())}."
                raise ValueError(msg)

        fig.supylabel("SHAP Value")

        # Add subplots
        for v, axis in zip(features, ax.flatten(), strict=False):
            color_feature = color_feature_dict.get(v)
            if color_feature is not None:
                color_axis = color_axis_dict[color_feature]
            else:
                color_axis = {"values": color}

            x = xp.X[v]
            y = xp.shap_values[:, xp.feature_names.index(v)]

            is_numeric = pd.api.types.is_numeric_dtype(x)  # numeric or bool
            is_discrete = (not is_numeric) or (len(x.unique()) <= 7)
            has_nulls = x.isna().any()

            if not is_discrete:
                if has_nulls:
                    xmin, xmax = x.min(), x.max()
                    gap = min(1, (xmax - xmin) / 10)
                    fill_value = xmax + gap
                    x = x.fillna(fill_value)

                scatter = axis.scatter(
                    x,
                    y,
                    s=s,
                    c=color_axis["values"],
                    zorder=3,
                    cmap=color_axis.get("cmap"),
                    norm=color_axis.get("norm"),
                    plotnonfinite=True,
                    **kwargs,
                )

                # Show "NA" on x axis
                if has_nulls:
                    xticks = axis.get_xticks()
                    xticks = np.append(xticks[xticks <= xmax], [fill_value])
                    axis.set_xticks(xticks)
                    xticklabels = axis.get_xticklabels()
                    xticklabels[-1] = "nan"
                    axis.set_xticklabels(xticklabels)
            else:  # discrete case
                xpos = x.drop_duplicates().sort_values(na_position="last").astype(str)
                x = x.astype(str)
                npos = len(xpos)

                for i, x_val in enumerate(xpos):
                    mask = x == x_val
                    xi = i
                    yi = y[mask]
                    if jitter_width > 0:
                        xi += jitter_width * beeswarm_jitter(yi)
                    ci = color_axis.get("values")
                    if color_feature is not None:
                        ci = ci[mask]
                    scatter = axis.scatter(
                        xi,
                        yi,
                        s=s,
                        c=ci,
                        zorder=3,
                        cmap=color_axis.get("cmap"),
                        norm=color_axis.get("norm"),
                        plotnonfinite=True,
                        **kwargs,
                    )

                axis.set_xticks(range(npos))

                # Reduce number of x labels
                if npos > max_xlabels:
                    step = int(np.ceil(npos / max_xlabels))
                    for i in range(npos):
                        if 0 < i < npos - 1 and i % step > 0:
                            xpos.iloc[i] = ""
                    npos = max_xlabels

                xpos = xpos.str[:max_xlabel_length]  # Truncate long labels
                axis.set_xticklabels(xpos)

                # Rotate x-axis labels
                if not is_numeric and (
                    (rotate_xlabels is None and npos > 4)
                    or rotate_xlabels not in (None, 0)
                ):
                    if rotate_xlabels is None:
                        rot = 45 if npos < 10 else 90
                    else:
                        rot = rotate_xlabels

                    axis.set_xticklabels(
                        axis.get_xticklabels(),
                        rotation=rot,
                        ha="right",
                        va="center",
                        rotation_mode="anchor",
                    )

            # Add color bar
            if color_axis.get("cmap") is not None:
                cbar = fig.colorbar(scatter, ax=axis, pad=0.01, aspect=30)
                cbar.ax.tick_params(size=0, pad=2, labelsize=9.0, which="both")

                mapping = color_axis.get("mapping")
                if mapping is not None:
                    ticks = list(mapping.keys())
                    labels = list(mapping.values())
                    cbar.set_ticks(ticks, labels=labels)
                    cbar.ax.set_title(
                        color_feature_dict[v], loc="left", y=1.0, fontsize=10.0
                    )
                else:
                    cbar.ax.set_ylabel(color_feature_dict[v])

                cbar.outline.set_visible(False)

            # Layout
            axis.set_xlabel(v)
            axis.grid(alpha=0.3, zorder=0)

        # Remove empty subplots (only if no ax object was passed by user)
        if cleanup:
            [fig.delaxes(axis) for axis in ax.flatten() if not axis.has_data()]

        # If user has passed a single axis, let's return a single axis again
        if single_axis_as_array:
            ax = ax[0]

        return ax

    def waterfall(
        self,
        row_id=0,
        max_display=10,
        fill_colors=("#b40426", "#3b4cc0"),
        annotation=("E[f(x)]", "f(x)"),
        fontsize=11.0,
        max_label_length=20,
        ax=None,
        which_output=None,
        **kwargs,
    ):
        """
        Waterfall plot visualizing the SHAP values of a single observation.

        Parameters
        ----------
        row_id : int, optional
            Row index of the observation to plot. Default is 0.
        max_display : int, optional
            Maximum number of features to display. If there are more features,
            they will be collapsed into "m other features". Default is 10.
            Set to None to show all features.
        fill_colors : tuple or list of str, optional
            Colors for positive and negative SHAP values.
        annotation : tuple or list of str, optional
            Annotations to show on the plot. Default is ("E[f(x)]", "f(x)").
            None to suppress annotation.
        fontsize : float, optional
            Font size for all text elements. Default is 11.0.
        max_label_length : int, optional
            Maximum length of non-numeric feature labels. Default is 20.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axis to use. If None, a new figure is created.
        which_output : int or string, optional
            Index or name of the output dimension to plot.
            If None, the last output is plotted. Only relevant for multi-output models.
        **kwargs : keyword arguments
            Additional keyword arguments to pass to ax.text().

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes containing the waterfall plot.

        Example
        -------
        >>> import matplotlib.pyplot as plt
        >>> explainer = Explanation(...)
        >>> explainer.plot.waterfall(row_id=17)
        >>> plt.savefig("filename.png", bbox_inches="tight")
        """
        xp = self._explanation

        # Pick output dimension
        if xp.ndim == 3:
            if which_output is None:
                which_output = xp.shape[2] - 1  # pick last
                print("Selected last output. Use which_output to use a different one.")
            xp = xp.select_output(which_output)

        if not isinstance(row_id, int) or not (0 <= row_id < len(xp)):
            msg = f"row_id {row_id} must be integer and < {len(xp)} observations"
            raise ValueError(msg)

        if not isinstance(fill_colors, list | tuple) or len(fill_colors) != 2:
            msg = "fill_colors must be a list or tuple of length 2"
            raise ValueError(msg)

        if annotation is not None and (
            not isinstance(annotation, list | tuple) or len(annotation) != 2
        ):
            msg = "annotation must be a list or tuple of length 2, or None."
            raise ValueError(msg)

        # y axis labels (might be simplified)
        labels = []
        for feature in xp.feature_names:
            value = xp.X.iloc[row_id][feature]
            if pd.isna(value):
                to_display = "nan"
            elif isinstance(value, int | float):
                to_display = f"{value:.3g}"
            else:
                to_display = str(value)
                if len(to_display) > max_label_length:
                    to_display = to_display[:max_label_length]
            labels.append(f"{feature} = {to_display}")

        # Create DataFrame with SHAP values and feature labels
        df = pd.DataFrame(
            {
                "shap": xp.shap_values[row_id],
                "width": np.abs(xp.shap_values[row_id]),
                "label": labels,
            }
        )
        n = df.shape[0]

        # Collapse SHAP values of other features
        if max_display is not None and max_display < n:
            # Keep top
            df = df.sort_values("width", ascending=False)
            keep_features = df.iloc[: max_display - 1]
            other_features = df.iloc[max_display - 1 :]

            # Collapse rest
            other_shap = other_features["shap"].sum()
            other_entry = pd.DataFrame(
                {
                    "shap": other_shap,
                    "width": np.abs(other_shap),
                    "label": f"{other_features.shape[0]} other features",
                },
                index=[0],
            )
            df = pd.concat([keep_features, other_entry], ignore_index=True)
            n = df.shape[0]

        # Order dependent calculations
        baseline = xp.baseline[0]
        df = df.sort_values("width", ascending=True).reset_index(drop=True)
        df = df.assign(
            From=lambda x: baseline + x.shap.cumsum() - x.shap,
            To=lambda x: baseline + x.shap.cumsum(),
            right=lambda x: x.To >= x.From,
            pos_left=lambda x: np.where(x.right, x.From, x.To),
            pos_right=lambda x: np.where(x.right, x.To, x.From),
            fill_color=lambda x: np.where(x.right, fill_colors[0], fill_colors[1]),
        )

        # Maximal arrowhead width (we need x_min, x_max below)
        x_min = df[["From", "To"]].min(axis=None)
        x_max = df[["From", "To"]].max(axis=None)
        x_range = x_max - x_min
        arrowhead_width_max = 0.04 * x_range

        # We make bars a little bit less wide to have space for arrowheads
        df = df.assign(
            arrowhead_width=lambda x: np.minimum(arrowhead_width_max, x["width"]),
            pos_left_bar=lambda x: np.where(
                x.right,
                x.pos_left,
                np.minimum(x["pos_left"] + x["arrowhead_width"], x["pos_right"]),
            ),
            pos_right_bar=lambda x: np.where(
                x.right,
                np.maximum(x["pos_right"] - x["arrowhead_width"], x["pos_left"]),
                x["pos_right"],
            ),
            bar_width=lambda x: x["pos_right_bar"] - x["pos_left_bar"],
        )

        # ===========================================================================
        # Plot
        # ===========================================================================

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 2 + 0.6 * n))
        elif not isinstance(ax, plt.Axes):
            msg = "ax must be a matplotlib Axes."
            raise TypeError(msg)
        else:
            fig = ax.get_figure()

        half_bar_height = 0.3
        texts = []
        polygons = []

        # Arrows and bar texts
        for i, row in df.iterrows():
            if row["right"]:
                vertices = [
                    (row["pos_right_bar"], i + half_bar_height),
                    (row["pos_left_bar"], i + half_bar_height),
                    (row["pos_left_bar"], i - half_bar_height),
                    (row["pos_right_bar"], i - half_bar_height),
                    (row["pos_right"], i),
                ]
            else:
                vertices = [
                    (row["pos_left_bar"], i + half_bar_height),
                    (row["pos_right_bar"], i + half_bar_height),
                    (row["pos_right_bar"], i - half_bar_height),
                    (row["pos_left_bar"], i - half_bar_height),
                    (row["pos_left"], i),
                ]
            polygon = Polygon(vertices, closed=True, color=row["fill_color"], zorder=3)
            ax.add_patch(polygon)
            polygons.append(polygon)

            text_x = (row["pos_left_bar"] + row["pos_right_bar"]) / 2
            text = ax.text(
                text_x,
                i,
                f"{row['shap']:+.3g}",
                ha="center",
                va="center",
                fontsize=fontsize,
                color="white",
                **kwargs,
            )
            texts.append(text)

        # Connections between bars
        if n > 1:
            for i in range(n - 1):
                xi = [df["To"][i], df["To"][i]]
                yi = [i + 0.1, i + 1 - half_bar_height - 0.1]
                ax.plot(xi, yi, color="gray", linestyle="--", linewidth=1)

        # Annotations
        if annotation is not None:
            arrowprops = {"arrowstyle": "->", "color": "gray"}

            # E[f(x)]
            ax.annotate(
                f"{annotation[0]} = {baseline:.3g}",
                xy=(baseline, -0.35),
                xytext=(baseline, -0.95),
                ha="center",
                va="center",
                arrowprops=arrowprops,
                fontsize=fontsize,
            )

            # f(x)
            prediction = df["To"].iloc[-1]

            ax.annotate(
                f"{annotation[1]} = {prediction:.3g}",
                xy=(prediction, n - 0.9),
                xytext=(prediction, n - 0.3),
                ha="center",
                va="center",
                arrowprops=arrowprops,
                fontsize=fontsize,
            )

        # Layout
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_yticks(range(n), labels=df["label"])
        ax.set_ylim(-0.5, n - 0.5)

        # Extend x-axis limits to show white space on both sides
        if annotation is not None:
            padh = 0.03 * (x_max - x_min)
            padv = 0.1
            text_left, text_right, text_bottom, text_top = get_text_bbox(ax)
            (left, right), (bottom, top) = ax.get_xlim(), ax.get_ylim()
            ax.set_xlim(min(left, text_left) - padh, max(text_right, right) + padh)
            ax.set_ylim(min(bottom, text_bottom) - padv, max(text_top, top) + padv)

        ax.tick_params(axis="both", labelsize=fontsize)

        # Remove too large texts from bars. To get correct pixel sizes, we move to end
        renderer = fig.canvas.get_renderer()
        for i, row in df.iterrows():
            text_width = texts[i].get_window_extent(renderer=renderer).width
            arrow_width = polygons[i].get_window_extent(renderer=renderer).width
            bar_width = arrow_width * row["bar_width"] / row["width"]

            if text_width > bar_width * 0.95:
                texts[i].remove()

        return ax
