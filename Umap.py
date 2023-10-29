import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import os

N = 2000
time_sample_length = 100
l0 = 0.01
lf = 15
tf = 1

lc = np.linspace(l0, lf, N)
t = np.linspace(0, tf, time_sample_length)

path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_PCA"
save_path = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Plots\\UMAP\\"
save_path_data = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_UMAP\\"
data_signals = pd.read_csv(os.path.join(path_read, "df_PCA_Signals.csv"))
data_probd = pd.read_csv(os.path.join(path_read, "df_PCA_Probd_100var.csv")).values.transpose()

lcms = data_signals["lcm"].values
sigmas = data_signals["σs"].values

# print(data_signals)

# signals_data = data_signals[
#     [
#         "pc1",
#         "pc2",
#     ]
# ].values

# print(signals_data.shape)

# print(data_probd.shape)

# print(data_probd)


#data_signals = data_signals.drop(["lcm", "σs"], axis=1).values

#UMAP parameters
n_components = 2
min_dist_list = [0, 0.05, 0.1, 0.5, 1]
n_neighbors_list = [5, 15, 30, 50, 100]
#data_signals = data_signals.drop(["lcm", "σs"], axis=1).values

embedding_OUT_plots = []

for i in range(len(n_neighbors_list)):
    for j in range(len(min_dist_list)):
        n_neighbors = n_neighbors_list[i]
        min_dist = min_dist_list[j]

        embeddingOUT = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components).fit_transform(data_probd)

        df_UMAPprobd = pd.DataFrame({
            'proyX': embeddingOUT[:, 0],
            'proyY': embeddingOUT[:, 1],
            'σs': sigmas,
            'lcm': lcms,
        })

        plot_UMAPprobd = df_UMAPprobd.plot.scatter(
            x='proyX',
            y='proyY',
            c='lcm',
            cmap='viridis',
            legend=False,
            title=f"min dist: {min_dist} n_neighbors: {n_neighbors}",
            figsize=(8, 6)
        )
        
        plot_UMAPprobd.figure.savefig(os.path.join(save_path, f"UMAP_min_dist_{min_dist}_n_neighbors_{n_neighbors}.png"))
        df_UMAPprobd.to_csv(os.path.join(save_path_data, f"df_UMAP_Probd_mdist_{min_dist}_nn_{n_neighbors}_nc_{n_components}.csv"), index=False)


# min_dist_list = [0.1, 0.5, 1]
# n_neighbors_list = [5, 15, 30, 50, 100]

# for i in range(len(n_neighbors_list)):
#     for j in range(len(min_dist_list)):
#         n_neighbors = n_neighbors_list[i]
#         min_dist = min_dist_list[j]

#         embeddingOUT = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components).fit_transform(data_probd)

        # df_UMAPprobd = pd.DataFrame({
        #     'σs': sigmas,
        #     'lcm': lcms,
        #     'proyX': embeddingOUT[:, 0],
        #     'proyY': embeddingOUT[:, 1]
        # })

#         plot_UMAPprobd = df_UMAPprobd.plot.scatter(
#             x='proyX',
#             y='proyY',
#             c='lcm',
#             cmap='viridis',
#             legend=False,
#             title=f"min dist: {min_dist}",
#             figsize=(8, 6)
#         )
        
#         plot_UMAPprobd.figure.savefig(os.path.join(save_path, f"UMAP_min_dist_{min_dist}_n_neighbors_{n_neighbors}.png"))
#         plt.close(plot_UMAPprobd.figure)

# embeddingOUT = umap.UMAP(n_neighbors=30, min_dist=0.5, n_components=n_components).fit_transform(data_probd)

# print(embeddingOUT.shape)

# df_UMAPprobd = pd.DataFrame({
#     'sigmas': sigmas,
#     'lcm': lcms,
#     'proyX': embeddingOUT[:, 0],
#     'proyY': embeddingOUT[:, 1]
# })

# plot_UMAPprobd = df_UMAPprobd.plot.scatter(
#     x='proyX',
#     y='proyY',
#     c='lcm',
#     cmap='viridis',
#     legend=False,
#     title=f"min dist: {0.5}",
#     figsize=(8, 6)
# )

# # scatter = plt.scatter(
# #     x=df_UMAPprobd['proyX'],
# #     y=df_UMAPprobd['proyY'],
# #     c=df_UMAPprobd['lcm'],
# #     cmap='viridis',
# #     s=df_UMAPprobd['sigmas'] * 100,  # Adjust the scaling factor as needed
# #     alpha=0.7  # Set marker transparency (optional)
# # )

# # # Add color bar and labels
# # cbar = plt.colorbar(scatter)
# # cbar.set_label('sigmas', rotation=270, labelpad=20)
# # plt.xlabel('proyX')
# # plt.ylabel('proyY')
# # plt.title("UMAP Probd")
# # plt.tight_layout()

# # plt.show()

# plt.savefig("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Plots\\UMAP\\UMAP_Probd_py.png")

# # # Create a subplot for UMAP Probd
# # fig, axs = plt.subplots(4, 3, figsize=(12, 16))
# # for i in range(len(embedding_OUT_plots)):
# #     plot_index = (i // 3, i % 3)
# #     axs[plot_index].remove()
# #     axs[plot_index] = fig.add_subplot(4, 3, i + 1)
# #     embedding_OUT_plots[i].get_figure().sca(axs[plot_index])

# # plt.tight_layout()
# # plt.savefig("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Plots\\UMAP\\UMAP_Probd_py.png")
# # embedding_IN_plots = []
# # embedding_OUT_plots = []

# # # UMAP Señales
# # for i in range(len(n_neighbors_list)):
# #     for j in range(len(min_dist_list)):
# #         n_neighbors = n_neighbors_list[i]
# #         min_dist = min_dist_list[j]
        
# #         embeddingIN = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components).fit_transform(data_signals)

# #         df_UMAPsignals = pd.DataFrame({
# #             'σs': lcms,
# #             'lcm': sigmas,
# #             'proyX': embeddingIN[:, 0],
# #             'proyY': embeddingIN[:, 1]
# #         })

# #         plot_UMAPsignals = df_UMAPsignals.plot.scatter(
# #             x='proyX',
# #             y='proyY',
# #             c='lcm',
# #             cmap='viridis',
# #             marker=(0.4, 5),
# #             legend=False,
# #             title=f"min dist: {min_dist}",
# #             figsize=(8, 6)
# #         )
# #         embedding_IN_plots.append(plot_UMAPsignals)

# # # Create a subplot for UMAP Señales
# # fig, axs = plt.subplots(4, 3, figsize=(12, 16))
# # for i in range(len(embedding_IN_plots)):
# #     plot_index = (i // 3, i % 3)
# #     axs[plot_index].remove()
# #     axs[plot_index] = fig.add_subplot(4, 3, i + 1)
# #     embedding_IN_plots[i].get_figure().sca(axs[plot_index])

# # plt.tight_layout()
# # plt.savefig("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Plots\\UMAP\\UMAP_Signals_py.png")

