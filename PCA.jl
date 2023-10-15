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
time_sample_lenght = 200

# Rango de tamaños de compartimientos en μm
l0 = 0.005
lf = 15

# Tiempo final de simulación en s
tf = 1

# Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
lc = range(l0, lf, length = N)
t = range(0, tf, length = time_sample_lenght)

# Parametros que se varian

# Rango de tamaños medios de correlación en μm
lcms = 0.5:0.01:6
σs = 0.01:0.05:1

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

path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Little_Data\\Little_Data_CSV"

dataSignals = GetSignals(path_read)
dataProbd = GetProbd(path_read)

#------------------------------------------------------------------------------------------

# Función que centra los datos de las columnas de una matriz

function CenterData(Non_C_Matrix)
	data_matrix = Non_C_Matrix
	col_means = mean(data_matrix, dims = 1)
	centered_data = data_matrix .- col_means
	return centered_data
end

# PCA para ciertos datos

function PCA_Data(dataIN)

    dataIN_C = CenterData(dataIN)

    # Esto ya hace PCA sobre la matriz dada donde cada observación es una columna de la matriz
    pca_model = fit(PCA, dataIN_C)

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

reduced_data_Signals = PCA_Data(dataSignals)[1]
reduced_data_Probd = PCA_Data(dataProbd)[1]

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
        pc2 = reduced_data_Probd[2, :],
        σs = column_σs,
        lcm = column_lcm,
    )

# Guardamos estos datos en CSV

path_save = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Little_Data\\Little_Data_CSV"

CSV.write(path_save * "\\df_PCA_Signals.csv", df_PCA_Signals)
CSV.write(path_save * "\\df_PCA_Probd.csv", df_PCA_Probd)

#------------------------------------------------------------------------------------------


