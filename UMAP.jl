using MultivariateStats
using DataFrames
using CSV
using Statistics
using UMAP
using StatsPlots
using LaTeXStrings
using Distances

#------------------------------------------------------------------------------------------
# Traemos los mismos datos de los parametros utilizados para generar los datos, deberiamos hacer una función grande en la proxima función que genere los datos donde les pasamos
# Todos estos parámetros desde otro programa, como ahora generamos pocos datos me quedo con esto

###################################################ACORDARSE DE ESTO#################################################################

# Parámetros fijos

# Lo que dejamos constante es el número de compartimientos, el rango de tamaños de correlación lc, el tiempo de simulación final y el muestreo de timepos
N = 2000
time_sample_lenght = 100

# Rango de tamaños de compartimientos en μm
l0 = 0.01
lf = 15

# Tiempo final de simulación en s
tf = 1

# Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
lc = range(l0, lf, length = N)
t = range(0, tf, length = time_sample_lenght)

# Parametros que se varian

# Rango de tamaños medios de correlación en μm
lcms = 0.5:0.01:6
σs = 0.01:0.01:1

#------------------------------------------------------------------------------------------
# Leemos los datos a los que les realizamos PCA

path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_PCA"

dataSignals = CSV.read(path_read * "\\df_PCA_Signals.csv", DataFrame)
dataSignals = Matrix(dataSignals)

dataProbd = CSV.read(path_read * "\\df_PCA_Probd.csv", DataFrame)
dataProbd = Matrix(dataProbd)

#------------------------------------------------------------------------------------------
# Hagamoslo con los datos originales de las señales y las distribuciones de probabilidad

# path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\DatosCSV"

# dataSignals = CSV.read(path_read * "\\dataSignals.csv", DataFrame)
# dataSignals = Matrix(dataSignals)

# dataProbd = CSV.read(path_read * "\\dataProbd.csv", DataFrame)
# dataProbd = Matrix(dataProbd)

#------------------------------------------------------------------------------------------

# Identificación de los datos reducidos con señales y distribuciones de probabilidad originales

dim1 = dimlcm = length(lcms)
dim2 = dimσ = length(σs)

column_lcm = zeros(dim1*dim2)
column_σs = zeros(dim1*dim2)
aux_lcm = collect(lcms)
aux_σs = collect(σs)

for i in 1:dim1
    for j in 1:dim2
        column_lcm[(i - 1)*dim2 + j] = aux_lcm[i]
        column_σs[(i - 1)*dim2 + j] = aux_σs[j]
    end
end

#------------------------------------------------------------------------------------------
# Exploración de UMAP reduciendo a 2 dimensiones

n_components = 2
min_dist_list = [0.1, 0.5, 1]
n_neighbors_list = [5, 15, 30, 100]#, 15, 30, 50, 100]

embedding_IN_plots = []
embedding_OUT_plots = []

#################### UMAP Señales

for i in 1:length(n_neighbors_list)
    for j in 1:length(min_dist_list)
        n_neighbors = n_neighbors_list[i]
        min_dist = min_dist_list[j]

        embeddingIN = umap(dataSignals, n_components; n_neighbors, min_dist)

        df_UMAPsignals = DataFrame(
                        σs = column_lcm,
                        lcm = column_σs,
                        proyX = embeddingIN[1, :],
                        proyY = embeddingIN[2, :],
                        )
        
        if (i == 1 && j == 1)
            plot_UMAPsignals = @df df_UMAPsignals  StatsPlots.scatter(
                :proyX,
                :proyY,
                group = :lcm,
                marker = (0.4,5),
                labels = false,
                xaxis = (title = ""),
                yaxis = (title = "N-N = $(n_neighbors_list[i])"),
                xlabel = "",
                ylabel = "N-N = $(n_neighbors_list[i])",
                title = "min dist: $(min_dist_list[j])",
            )
        end

        if (i == 1 && j != 1)
            plot_UMAPsignals = @df df_UMAPsignals  StatsPlots.scatter(
                :proyX,
                :proyY,
                group = :lcm,
                marker = (0.4,5),
                labels = false,
                xaxis = (title = ""),
                yaxis = (title = ""),
                xlabel = "",
                ylabel = "",
                title = "min dist: $(min_dist_list[j])",
            )
        end

        if (j == 1 && i!=1)
            plot_UMAPsignals = @df df_UMAPsignals StatsPlots.scatter(
                :proyX,
                :proyY,
                group = :lcm,
                marker = (0.4,5),
                labels = false,
                xaxis = (title = ""),
                yaxis = (title = "N-N = $(n_neighbors_list[i])"),
                xlabel = "",
                ylabel = "N-N = $(n_neighbors_list[i])",
                title = "",
            )
        end

        if (i != 1 && j != 1)
            plot_UMAPsignals = @df df_UMAPsignals StatsPlots.scatter(
                :proyX,
                :proyY,
                group = :lcm,
                marker = (0.4,5),
                xaxis = (title = ""),
                yaxis = (title = ""),
                labels = false,
                xlabel = "",
                ylabel = "",
                title = "",
            )
        end
        push!(embedding_IN_plots, plot_UMAPsignals)
    end
end

pls = Plots.plot(embedding_IN_plots[1], embedding_IN_plots[2], embedding_IN_plots[3], 
                embedding_IN_plots[4], embedding_IN_plots[5], embedding_IN_plots[6],
                embedding_IN_plots[7], embedding_IN_plots[8], embedding_IN_plots[9],
                embedding_IN_plots[10], embedding_IN_plots[11], embedding_IN_plots[12],
                layout = (4,3), size = (1100,1000))

savefig(pls, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Plots\\UMAP\\UMAP_Signals_o.png")

####################### UMAP Probd

for i in 1:length(n_neighbors_list)
    for j in 1:length(min_dist_list)
        n_neighbors = n_neighbors_list[i]
        min_dist = min_dist_list[j]

        embeddingOUT = umap(dataProbd, n_components; n_neighbors, min_dist)

        df_UMAPprobd = DataFrame(
                        σs = column_lcm,
                        lcm = column_σs,
                        proyX = embeddingOUT[1, :],
                        proyY = embeddingOUT[2, :],
                        )
        
        if (i == 1 && j == 1)
            plot_UMAPprobd = @df df_UMAPprobd  StatsPlots.scatter(
                :proyX,
                :proyY,
                group = :lcm,
                marker = (0.4,5),
                labels = false,
                xaxis = (title = ""),
                yaxis = (title = "N-N = $(n_neighbors_list[i])"),
                xlabel = "",
                ylabel = "N-N = $(n_neighbors_list[i])",
                title = "min dist: $(min_dist_list[j])",
            )
        end

        if (i == 1 && j != 1)
            plot_UMAPprobd = @df df_UMAPprobd  StatsPlots.scatter(
                :proyX,
                :proyY,
                group = :lcm,
                marker = (0.4,5),
                labels = false,
                xaxis = (title = ""),
                yaxis = (title = ""),
                xlabel = "",
                ylabel = "",
                title = "min dist: $(min_dist_list[j])",
            )
        end

        if (j == 1 && i!=1)
            plot_UMAPprobd = @df df_UMAPprobd StatsPlots.scatter(
                :proyX,
                :proyY,
                group = :lcm,
                marker = (0.4,5),
                labels = false,
                xaxis = (title = ""),
                yaxis = (title = "N-N = $(n_neighbors_list[i])"),
                xlabel = "",
                ylabel = "N-N = $(n_neighbors_list[i])",
                title = "",
            )
        end

        if (i != 1 && j != 1)
            plot_UMAPprobd = @df df_UMAPprobd StatsPlots.scatter(
                :proyX,
                :proyY,
                group = :lcm,
                marker = (0.4,5),
                xaxis = (title = ""),
                yaxis = (title = ""),
                labels = false,
                xlabel = "",
                ylabel = "",
                title = "",
            )
        end
        push!(embedding_OUT_plots, plot_UMAPprobd)
    end
end


pl = Plots.plot(embedding_OUT_plots[1], embedding_OUT_plots[2], embedding_OUT_plots[3], 
                embedding_OUT_plots[4], embedding_OUT_plots[5], embedding_OUT_plots[6],
                embedding_OUT_plots[7], embedding_OUT_plots[8], embedding_OUT_plots[9],
                embedding_OUT_plots[10], embedding_OUT_plots[11], embedding_OUT_plots[12],
                layout = (4,3), size = (1100,1000))

savefig(pl, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Plots\\UMAP\\UMAP_Probd_.png")