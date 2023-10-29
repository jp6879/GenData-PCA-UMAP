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

path_images = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Plots\\UMAP\\"
path_data = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_UMAP\\"

dataframes = []
min_dist_list = [0, 0.05, 0.1, 0.5, 1]
n_neighbors_list = [5, 15, 30, 50, 100]

for min_dist in min_dist_list:
    for n_neighbour in n_neighbors_list:
        dataframe = pd.read_csv(os.path.join(path_data, f"df_UMAP_Probd_mdist_{min_dist}_nn_{n_neighbour}_nc_2.csv"))
        dataframes.append(dataframe)


# Plot all the dataframes in the same plot
fig, ax = plt.subplots(5, 5, figsize=(20, 20))
for i, min_dist in enumerate(min_dist_list):
    for j, n_neighbour in enumerate(n_neighbors_list):
        df = dataframes[i*5 + j]
        ax[i, j].scatter(df["proyX"], df["proyY"], c=df["lcm"], s=df["σs"]*10, cmap="viridis")
        if(i == 0):
           ax[i, j].set_title(f"n_neighbour = {n_neighbors_list[j]}")
        if(j == 0):
            ax[i, j].set_ylabel(f"min_dist = {min_dist_list[i]}")

        # if(i == 0 and j == len(min_dist_list)-1):
        #     cbar = plt.colorbar(ax[i, j].collections[0], ax=ax[i, j], location="right", shrink=0.6, pad=0.1, aspect=10, ticks=[0, 5, 10, 15], label="lcm")
        #ax[i, j].set_title(f"min_dist = {min_dist}, n_neighbour = {n_neighbour}")
plt.tight_layout()
plt.savefig(os.path.join(path_images, "UMAP_Probd.png"))
# plt.show()
plt.close()

