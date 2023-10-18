using Plots
using MultivariateStats
using DataFrames
using CSV
using Statistics
using StatsPlots

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

# Leemos los datos que generamos
#C:\Users\Propietario\Desktop\ib\5-Maestría\GenData-PCA-UMAP\Little_Data\Little_Data_CSV\dataSignals.csv
# Función que lee los datos de las señales
function GetSignals(path_read)
    dataSignals = CSV.read(path_read * "\\dataSignals.csv", DataFrame)
    dataSignals = Matrix(dataSignals)
    return dataSignals
end

# Función que lee los datos de las distribuciones de probabilidad
function GetProbd(path_read)
    dataProbd = CSV.read(path_read * "\\dataProbd.csv", DataFrame)
    dataProbd = Matrix(dataProbd)
    return dataProbd
end

#path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Little_Data\\Little_Data_CSV"
path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\DatosCSV"

dataSignals = GetSignals(path_read)
dataProbd = GetProbd(path_read)

#------------------------------------------------------------------------------------------

# PCA para ciertos datos

function PCA_Data(dataIN)

    dataIN_C = CenterData(dataIN)

    # Esto ya hace PCA sobre la matriz dada donde cada observación es una columna de la matriz
    pca_model = fit(PCA, dataIN_C; maxoutdim = 2)

    # Esta instancia de PCA tiene distintas funciones como las siguientes

    #projIN = projection(pca_model) # Proyección de los datos sobre los componentes principales

    # Vector con las contribuciones de cada componente (es decir los autovalores)
    pcsIN = principalvars(pca_model)

    # Obtenemos la variaza en porcentaje para cada componente principal
    explained_varianceIN = pcsIN / sum(pcsIN) * 100

    # Grafiquemos esto para ver que tan importante es cada componente principal
    display(Plots.bar(explained_varianceIN, title="Varianza en porcentaje datos entrada",label = false, xlabel="Componente principal", ylabel="Varianza (%)"))

    reduced_dataIN = MultivariateStats.transform(pca_model, dataIN_C)

    return reduced_dataIN, pca_model

end

#------------------------------------------------------------------------------------------

reduced_data_Signals, pca_model_signals = PCA_Data(dataSignals)
reduced_data_Probd, pca_model_probd = PCA_Data(dataProbd)

#------------------------------------------------------------------------------------------

# Quiero ver hasta que componente hay una varianza acumulada del 98% en las distribuciones de probabilidad
pcs_vars_s = principalvars(pca_model_signals)
pcs_vars_pd = principalvars(pca_model_probd)

limdim_S = 0
limdim_P = 0
for i in 1:length(pcs_vars_s)
    if sum(pcs_vars_s[1:i]) / sum(pcs_vars_s) * 100 > 98
        println("La varianza acumulada de las señales es del ", sum(pcs_vars_s[1:i]) / sum(pcs_vars_s) * 100, "% con ", i, " componentes principales")
        limdim_S = i
        break
    end
end

for i in 1:length(pcs_vars_pd)
    if sum(pcs_vars_pd[1:i]) / sum(pcs_vars_pd) * 100 > 98
        println("La varianza acumulada de las distribuciones de probabilidad es del ", sum(pcs_vars_pd[1:i]) / sum(pcs_vars_pd) * 100, "% con ", i, " componentes principales")
        limdim_P = i
        break
    end
end

df_PCA_Signals = DataFrame(reduced_data_Signals, :auto)
df_PCA_Probd = DataFrame(reduced_data_Probd, :auto)

df_PCA_Signals = df_PCA_Signals[1:limdim_S,:]
df_PCA_Probd = df_PCA_Probd[1:limdim_P,:]

#------------------------------------------------------------------------------------------
# Datos reconstruidos

#reconstruct(M::PCA, y::AbstractVecOrMat{<:Real})

re_signals = reconstruct(pca_model_signals, reduced_data_Signals)
re_probd = reconstruct(pca_model_probd, reduced_data_Probd)

Plots.scatter(t,re_signals[:,0*100 + 1], label = "lcm = $(lcms[1]), σ = $(σs[1])", markersize = 2)
Plots.scatter!(t,re_signals[:,0*100 + 20], label = "lcm = $(lcms[1]), σ = $(σs[20])", markersize = 2)
Plots.scatter!(t,re_signals[:,0*100 + 100], label = "lcm = $(lcms[1]), σ = $(σs[100])", markersize = 2)
Plots.scatter!(t,re_signals[:,(20 - 1)*100 + 1], label = "lcm = $(lcms[20]), σ = $(σs[1])", markersize = 2)
Plots.scatter!(t,re_signals[:,(20 - 1)*100 + 100], label = "lcm = $(lcms[20]), σ = $(σs[100])", markersize = 2)

Plots.scatter(lc,re_probd[:,(50-1)*100 + 20], label = "lcm = $(lcms[50]), σ = $(σs[20])", markersize = 0.5)
Plots.scatter!(lc,re_probd[:,(60-1)*100 + 100], label = "lcm = $(lcms[50]), σ = $(σs[100])", markersize = 0.5)
Plots.scatter!(lc,re_probd[:,(551 - 1)*100 + 1], label = "lcm = $(lcms[551]), σ = $(σs[1])", markersize = 0.5)
Plots.scatter!(lc,re_probd[:,(551 - 1)*100 + 100], label = "lcm = $(lcms[551]), σ = $(σs[100])", markersize = 0.001)



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

# Guardamos la identificacion y los datos transformados en un DataFrame para graficos, se podria tambien guardarlos en CSV

df_PCA_Signals = DataFrame(
		pc1 = reduced_data_Signals[1, :],
	    pc2 = reduced_data_Signals[2, :],
	    σs = column_σs,
	    lcm = column_lcm,
	)

df_PCA_Probd = DataFrame(
        pc1 = reduced_data_Probd[1, :],
        pc2 = -reduced_data_Probd[2, :],
        σs = column_σs,
        lcm = column_lcm,
    )


plot_lcms_S = @df df_PCA_Signals StatsPlots.scatter(
    :pc1,
    :pc2,
    group = :lcm,
    marker = (0.4,5),
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
    labels = false,  # Use the modified labels
    title = "PCA para S(t)",
)

plot_lcms_PD = @df df_PCA_Probd StatsPlots.scatter(
    :pc1,
    :pc2,
    group = :lcm,
    marker = (0.4,5),
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
    labels = false,  # Use the modified labels
    title = "PCA para P(lc)"
)

# Guardamos estos datos en CSV

#path_save = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Little_Data\\Little_Data_CSV"
path_save = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_PCA"

CSV.write(path_save * "\\df_PCA_Signals.csv", df_PCA_Signals)
CSV.write(path_save * "\\df_PCA_Probd.csv", df_PCA_Probd)

#------------------------------------------------------------------------------------------


