import os
import pandas as pd
import seaborn as sns
import numpy as np
from nilearn.connectome import vec_to_sym_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
import textwrap


def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        labels.append(
            textwrap.fill(text, width=width, break_long_words=break_long_words)
        )
    ax.set_yticklabels(labels, rotation=0)


def get_lower_tri_heatmap(
    df,
    figsize=(11, 9),
    cmap="viridis",
    annot=False,
    title=None,
    ticks=None,
    labels=None,
    grid=False,
    output="matrix.png",
    triu=False,
    diag=False,
    tril=False,
    fontsize=20,
    vmax=None,
    vmin=None,
    fontweight="normal",
):
    mask = np.zeros_like(df)
    mask[np.triu_indices_from(mask)] = triu

    mask[np.tril_indices_from(mask)] = tril

    # Want diagonal elements as well
    mask[np.diag_indices_from(mask)] = diag

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns_plot = sns.heatmap(
        df,
        mask=mask,
        cmap=cmap,
        # cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        # center=0,
        square=True,
        linewidths=0,
        cbar_kws={"shrink": 0.5},
        ax=ax,
        annot=annot,
        fmt="",
        # annot_kws={
        #     "backgroundcolor": "white",
        #     "color": "black",
        #     "bbox": {
        #         "alpha": 0.5,
        #         "color": "white",
        #     },
        # },
    )
    if grid:
        ax.grid(grid, color="black", linewidth=0.5)
    else:
        ax.grid(grid)
    if labels is not None and ticks is None:
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=fontsize)
        ax.set_yticklabels(labels, rotation=0, fontsize=fontsize)
    elif labels is not None and ticks is not None:
        ax.set_xticks(
            ticks, labels, fontsize=fontsize, rotation=40, ha="right"
        )
        ax.set_yticks(ticks, labels, fontsize=fontsize, rotation=45)
        ax.tick_params(left=True, bottom=True)
    else:
        ax.set_xticklabels([], fontsize=fontsize)
        ax.set_yticklabels([], fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize, fontweight=fontweight)

    # save to file
    fig = sns_plot.get_figure()
    fig.savefig(f"{output}.png", bbox_inches="tight")
    fig.savefig(f"{output}.svg", bbox_inches="tight", transparent=True)
    plt.close(fig)


def get_clas_cov_measure(classify, cov_estimators, measures):
    for clas in classify:
        for cov in cov_estimators:
            for measure in measures:
                yield clas, cov, measure


def get_network_labels(atlas):
    networks = atlas["labels"].astype("U")
    hemi_network_labels = []
    network_labels = []
    rename_networks = {
        "Vis": "Visual",
        "Cont": "FrontPar",
        "SalVentAttn": "VentAttn",
    }
    for network in networks:
        components = network.split("_")
        components[2] = rename_networks.get(components[2], components[2])
        hemi_network = " ".join(components[1:3])
        hemi_network_labels.append(hemi_network)
        network_labels.append(components[2])

    return hemi_network_labels, network_labels


def _load_transform_weights(
    clas, cov, measure, transform, weight_dir, n_parcels
):
    try:
        weights = np.load(
            os.path.join(weight_dir, f"{clas}_{cov} {measure}_weights.npy")
        )
    except FileNotFoundError:
        print(f"skipping {clas} {cov} {measure}")
        raise FileNotFoundError

    if transform == "maxratio":
        weights = np.abs(weights)
        max_weights = np.max(weights, axis=0)
        mask = weights.max(axis=0, keepdims=True) == weights
        mean_other_values = np.mean(weights[~mask], axis=0)
        weights = max_weights / mean_other_values
    elif transform == "l2":
        weights = np.sqrt(np.sum(weights**2, axis=0))

    weight_mat = vec_to_sym_matrix(weights, diagonal=np.ones(n_parcels))

    return weight_mat


def _average_over_networks(
    encoded_labels,
    unique_labels,
    clas,
    cov,
    measure,
    transform,
    weight_dir,
    n_parcels,
):
    network_pair_weights = np.zeros((len(unique_labels), len(unique_labels)))
    # get network pair weights
    for network_i in unique_labels:
        index_i = np.where(encoded_labels == network_i)[0]
        for network_j in unique_labels:
            index_j = np.where(encoded_labels == network_j)[0]
            weight_mat = _load_transform_weights(
                clas, cov, measure, transform, weight_dir, n_parcels
            )
            weight_mat[np.triu_indices_from(weight_mat)] = np.nan
            network_pair_weight = np.nanmean(
                weight_mat[np.ix_(index_i, index_j)]
            )
            network_pair_weights[network_i][network_j] = network_pair_weight

    return network_pair_weights


def plot_network_weight_matrix(
    clas,
    cov,
    measure,
    atlas,
    output_dir,
    weight_dir,
    n_parcels,
    labels_fmt="hemi network",
    transform="maxratio",
    fontsize=20,
):
    if labels_fmt == "hemi network":
        labels = get_network_labels(atlas)[0]
    elif labels_fmt == "network":
        labels = get_network_labels(atlas)[1]

    le = preprocessing.LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    unique_labels = np.unique(encoded_labels)
    network_pair_weights = _average_over_networks(
        encoded_labels,
        unique_labels,
        clas,
        cov,
        measure,
        transform,
        weight_dir,
        n_parcels,
    )
    # plot network wise average weights
    get_lower_tri_heatmap(
        network_pair_weights,
        figsize=(5, 5),
        cmap="viridis",
        labels=le.inverse_transform(unique_labels),
        output=os.path.join(
            output_dir, f"{clas}_{cov}_{measure}_network_weights"
        ),
        triu=True,
        title=f"{cov} {measure}",
        fontsize=fontsize,
    )


def plot_full_weight_matrix(
    clas,
    cov,
    measure,
    atlas,
    output_dir,
    weight_dir,
    n_parcels,
    transform="maxratio",
    fontsize=20,
):
    weight_mat = _load_transform_weights(
        clas,
        cov,
        measure,
        transform=transform,
        weight_dir=weight_dir,
        n_parcels=n_parcels,
    )

    hemi_network_labels = get_network_labels(atlas)[0]

    # get tick locations
    ticks = []
    for i, label in enumerate(hemi_network_labels):
        if label != hemi_network_labels[i - 1]:
            ticks.append(i)

    # keep unique labels
    _, uniq_idx = np.unique(hemi_network_labels, return_index=True)
    hemi_network_labels = np.array(hemi_network_labels)[np.sort(uniq_idx)]

    # plot full matrix
    get_lower_tri_heatmap(
        weight_mat,
        cmap="viridis",
        labels=hemi_network_labels,
        output=os.path.join(output_dir, f"{clas}_{cov}_{measure}_all_weights"),
        triu=True,
        diag=True,
        title=f"{cov} {measure}",
        ticks=ticks,
        grid=True,
        fontsize=fontsize,
    )


def insert_stats_horizontal(
    ax, p_val, data, y_positions, x_offset_level=0, bar_height=0.8
):
    """
    Insert p-values from statistical tests into horizontal barplots.
    Creates a separate axis for statistical annotations to avoid affecting the main plot.

    Parameters:
    -----------
    ax : matplotlib axis
        The main axis with the barplot
    p_val : float
        The p-value from statistical test
    data : array-like
        Data values (not used in this version)
    y_positions : list
        [y1, y2] positions of the bars being compared
    x_offset_level : int
        Level of offset to avoid overlapping lines (0, 1, 2, ...)
    bar_height : float
        Height of bars (for proper spacing)
    """
    # Create a separate axis for statistical annotations only if it doesn't exist
    fig = ax.get_figure()

    # Check if stats axis already exists
    stats_ax = None
    for axis in fig.axes:
        if hasattr(axis, "_is_stats_axis"):
            stats_ax = axis
            break

    if stats_ax is None:
        # Get the position of the main axis
        main_pos = ax.get_position()

        # Adjust main axis to make room for stats axis
        new_main_width = main_pos.width * 0.10  # Reduce main axis width by 75%
        ax.set_position(
            [main_pos.x0, main_pos.y0, new_main_width, main_pos.height]
        )

        # Create an even wider axis on the right for statistics
        stats_width = 0.35  # Much larger width of the statistics axis
        gap = 0.79  # Larger gap between axes for clear separation
        stats_ax = fig.add_axes(
            [
                main_pos.x0
                + new_main_width
                + gap,  # Start after main axis + gap
                main_pos.y0 + 0.02,  # Same bottom position
                stats_width,  # Width for statistics
                main_pos.height - 0.03,  # Same height
            ]
        )

        # Mark this as a stats axis
        stats_ax._is_stats_axis = True

        # Set the same y-limits as the main axis
        stats_ax.set_ylim(ax.get_ylim())

        # Set an even wider x range for the statistics to accommodate all lines
        stats_ax.set_xlim(0, 35)  # Even larger range

        # Remove ticks and labels from the stats axis
        stats_ax.set_xticks([])
        stats_ax.set_yticks([])
        stats_ax.spines["top"].set_visible(False)
        stats_ax.spines["right"].set_visible(False)
        stats_ax.spines["bottom"].set_visible(False)
        stats_ax.spines["left"].set_visible(False)

    y1, y2 = y_positions[0], y_positions[1]

    # Draw lines in the statistics axis with better spacing
    line_x = 1 + (x_offset_level * 2.2)  # More spacing between levels
    line_length = 0.5  # Shortened from 1.5 to 0.8

    # Horizontal lines at each bar level (shorter lines)
    stats_ax.plot([line_x, line_x + line_length], [y1, y1], lw=1.5, c="0.25")
    stats_ax.plot([line_x, line_x + line_length], [y2, y2], lw=1.5, c="0.25")
    # Vertical connecting line
    stats_ax.plot(
        [line_x + line_length, line_x + line_length],
        [y1, y2],
        lw=1.5,
        c="0.25",
    )

    # Determine significance text
    if p_val < 0.0001:
        text = "****"
    elif p_val < 0.001:
        text = "***"
    elif p_val < 0.01:
        text = "**"
    elif p_val < 0.05:
        text = "*"
    else:
        text = "ns"

    # Position text at the middle of the line, with some spacing to the right
    text_x = line_x + line_length + 0.2  # Increased spacing from 0.4 to 0.8
    text_y = (y1 + y2) / 2

    stats_ax.text(
        text_x,
        text_y,
        f"{text}",
        ha="center",
        va="center",
        color="0.25",
        fontsize=11,
        weight="bold",
        rotation=270,  # Rotate 90 degrees to be parallel with vertical comparison lines
    )


def insert_stats(ax, p_val, data, loc=[], h=0.15, y_offset=0, x_n=3):
    """
    Insert p-values from statistical tests into boxplots.
    """
    max_y = data.max()
    h = h / 100 * max_y
    y_offset = y_offset / 100 * max_y
    x1, x2 = loc[0], loc[1]
    y = max_y + h + y_offset
    ax.plot([y, y], [x1, x2], lw=2, c="0.25")
    if p_val < 0.0001:
        text = "****"
    if p_val < 0.001:
        text = "***"
    elif p_val < 0.01:
        text = "**"
    elif p_val < 0.05:
        text = "*"
    else:
        text = "ns"
    ax.text(
        y + 3.5,
        ((x1 + x2) * 0.5) - 0.15,
        f"{text}",
        ha="center",
        va="bottom",
        color="0.25",
    )
    ax.set_xticks([*range(0, x_n)])
    ax.axis("off")


def mean_connectivity(data, tasks, cov_estimators, measures):
    """Average connectivity across runs for each subject and task.

    Parameters
    ----------
    data : pandas dataframe
        a dataframe with flattened connectivity matrices with a
        column for each fc measure (created by joining covariance
        estimator and the measure with a space), a column for
        the task, and a column for the subject
    tasks : list
        a list of tasks to average connectivity across runs
    cov_estimators : list
        a list of covariance estimators
    measures : list
        a list of connectivity measures estimated by each covariance

    Returns
    -------
    pandas dataframe
        a dataframe with the average connectivity for each subject,
        task, and measure in long format
    """
    av_connectivity = []
    for task in tasks:
        task_data = data[data["tasks"] == task]
        task_subjects = task_data["subject_ids"].unique()
        for sub in task_subjects:
            df = task_data[task_data["subject_ids"] == sub]
            for cov in cov_estimators:
                for measure in measures:
                    connectivity = df[cov + " " + measure].tolist()
                    connectivity = np.array(connectivity)
                    connectivity = connectivity.mean(axis=0)
                    av_connectivity.append(
                        {
                            "task": task,
                            "subject": sub,
                            "connectivity": connectivity,
                            "measure": cov + " " + measure,
                        }
                    )

    return pd.DataFrame(av_connectivity)


def insert_stats_reliability(
    ax,
    p_val,
    data,
    loc=[],
    h=0.15,
    y_offset=0,
    x_n=3,
):
    """
    Insert p-values from statistical tests into boxplots.
    """
    h = h / 100 * data
    y_offset = y_offset / 100 * data
    x1, x2 = loc[0], loc[1]
    y = data + h + y_offset
    ax.plot([y, y], [x1, x2], lw=2, c="0.25")
    if p_val < 0.0001:
        text = "****"
    if p_val < 0.001:
        text = "***"
    elif p_val < 0.01:
        text = "**"
    elif p_val < 0.05:
        text = "*"
    else:
        text = "ns"
    ax.text(
        y + 3.5,
        ((x1 + x2) * 0.5) - 0.15,
        f"{text}",
        ha="center",
        va="bottom",
        color="0.25",
    )
    ax.set_xticks([*range(0, x_n)])
    ax.axis("off")
