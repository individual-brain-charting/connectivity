import matplotlib.pyplot as plt
import pandas as pd
import textwrap
import seaborn as sns

sns.set_context("talk")

# Load the compiled results
results = pd.read_pickle(
    "/Users/himanshu/Desktop/ibc/connectivity/results/permutation_tests/all_results.pkl"
)

column_order = {
    "Unregularized correlation": 0,
    "Ledoit-Wolf correlation": 1,
    "Graphical-Lasso partial correlation": 2,
}
movies_story = ["Raiders", "GoodBadUgly", "MonkeyKingdom", "LePetitPrince"]

# drop rows with classes "Runs" and task_label not in movies_story
results = results[
    (results["classes"] != "Runs") | (results["task_label"].isin(movies_story))
]
results.reset_index(inplace=True, drop=True)

# Create a figure and axis with subplots for each row
for scenario in ["Tasks", "Subjects", "Runs"]:
    df = results[results["classes"] == scenario]
    df.reset_index(inplace=True, drop=True)
    n_scenarios = df.shape[0]
    if n_scenarios > 0:
        print(f"{scenario}: {n_scenarios} scenarios")
    fig, axs = plt.subplots(
        n_scenarios // 3,
        3,
        figsize=(8 * 3, 4 * (n_scenarios // 3)),
        sharex=True,
        sharey=True,
    )
    # iterate over each connectivity measure
    for connectivity_measure in column_order.keys():
        df_ = df[df["connectivity_measure"] == connectivity_measure]
        df_.sort_values(by=["task_label"], inplace=True)
        df_.reset_index(inplace=True, drop=True)
        col_index = column_order[connectivity_measure]
        # iterate over rows of results
        for index, row in df_.iterrows():
            print(f"Row {index}")
            ax = axs[index - 1, col_index]
            if row["pvalue"] > 0.05:
                ax.hist(
                    row["permutation_scores"],
                    bins=20,
                    density=True,
                    color="red",
                )
            else:
                ax.hist(
                    row["permutation_scores"],
                    bins=20,
                    density=True,
                    color="green",
                )
            ax.axvline(row["scores"], ls="--", color="k")
            score_label = f"Original score: {row['scores']:.2f}\n(p: {row['pvalue']:.3f})"
            ax.text(0.6, 7, score_label, fontsize=20)
            # fix x axis range to 0-1
            ax.set_xlim(0, 1.1)
            # fix y axis range to 0-10
            ax.set_ylim(0, 10)
            if (
                len(row["task_label"].split("vs")) > 1
                and row["classes"] != "Tasks"
            ):
                classification_type = "Across tasks"
            elif (
                len(row["task_label"].split("vs")) == 2
                and row["classes"] == "Tasks"
            ):
                classification_type = f"Between {row['task_label'].split(' vs ')[0]} and {row['task_label'].split(' vs ')[1]}"
            elif (
                len(row["task_label"].split("vs")) == 1
                and row["classes"] != "Tasks"
            ):
                classification_type = f"Within {row['task_label']}"
            else:
                classification_type = "Across tasks"
            title = f"{row['classes']} ({classification_type}), {row['connectivity_measure']}"
            title = "\n".join(textwrap.wrap(title, 40))
            ax.set_title(title, fontsize=20)

    fig.supxlabel("F1 score", fontsize=20, y=-0.00001)
    fig.supylabel("Probability density", fontsize=20)
    plt.tight_layout()
    plt.savefig(
        f"/Users/himanshu/Desktop/ibc/connectivity/plots/permutation_tests/{scenario}_permutation_tests.png",
    )
    plt.savefig(
        f"/Users/himanshu/Desktop/ibc/connectivity/plots/permutation_tests/{scenario}_permutation_tests.svg",
        transparent=True,
    )
    plt.close()
