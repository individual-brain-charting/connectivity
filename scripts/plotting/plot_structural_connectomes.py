### structural connectivity plots ###

results_root = "/data/parietal/store3/work/haggarwa/connectivity/results/"
plots_root = (
    "/data/parietal/store3/work/haggarwa/connectivity/plots/wo_extra_GBU_runs"
)
n_parcels = 400
trim_length = None
sc_data = pd.read_pickle(
    os.path.join(results_root, f"sc_data_native_{n_parcels}")
)

mats_dir = os.path.join(plots_root, f"sc_matrices_n_parcels-{n_parcels}")
brain_dir = os.path.join(plots_root, f"sc_glass_brain_n_parcels-{n_parcels}")
html_dir = os.path.join(plots_root, f"sc_html_n_parcels-{n_parcels}")
for directory in [mats_dir, brain_dir, html_dir]:
    output_dir = os.path.join(plots_root, dir_name)
    os.makedirs(output_dir, exist_ok=True)
coords = pd.read_csv(
    f"Schaefer2018_{n_parcels}Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv"
)[["R", "A", "S"]].to_numpy()

# get atlas for yeo network labels
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, resolution_mm=2, n_rois=400
)
networks = atlas["labels"].astype("U")
hemi_network_labels = []
for network in networks:
    components = network.split("_")
    hemi_network = "_".join(components[1:3])
    hemi_network_labels.append(hemi_network)
ticks = []
for i, label in enumerate(hemi_network_labels):
    if label != hemi_network_labels[i - 1]:
        ticks.append(i)
# load the data
_, uniq_idx = np.unique(hemi_network_labels, return_index=True)
hemi_network_labels = np.array(hemi_network_labels)[np.sort(uniq_idx)]
sns.set_context("notebook", font_scale=1.05)

for sub in tqdm(np.unique(sc_data["subject"])):
    try:
        matrix = vec_to_sym_matrix(
            np.mean(
                np.vstack(
                    list(sc_data[sc_data["subject"] == sub]["connectivity"])
                ),
                axis=0,
            ),
            diagonal=np.ones(400),
        )
    except ValueError:
        print(f"{sub} does not exist")
        continue
    # plot connectome as a matrix
    get_lower_tri_heatmap(
        matrix,
        title=f"{sub}",
        output=os.path.join(mats_dir, f"{sub}"),
        ticks=ticks,
        labels=hemi_network_labels,
        grid=True,
        diag=True,
        triu=True,
    )
    f = plt.figure(figsize=(9, 4))
    # plot connectome on glass brain
    plot_connectome(
        matrix,
        coords,
        edge_threshold="99.8%",
        title=f"{sub}",
        node_size=25,
        figure=f,
        colorbar=True,
        output_file=os.path.join(
            brain_dir,
            f"{sub}_connectome.png",
        ),
    )
    threshold = np.percentile(matrix, 99.8)
    matrix_thresholded = np.where(matrix > threshold, matrix, 0)
    max_ = np.max(matrix)
    # plot connectome in 3D view in html
    three_d = view_connectome(
        matrix,
        coords,
        edge_threshold="99.8%",
        symmetric_cmap=False,
        title=f"{sub}",
    )
    three_d.save_as_html(
        os.path.join(
            html_dir,
            f"{sub}_connectome.html",
        )
    )
    plt.close("all")
