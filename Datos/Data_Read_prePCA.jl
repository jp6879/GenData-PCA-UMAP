# Acordarse de cambiar la extension de este archivo de $N a 2k
include("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Gen_Read_Data-Hahn.jl")
using CSV
using DataFrames

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

length(lcms) * length(σs)

#------------------------------------------------------------------------------------------
# Generación de datos en CSV para cada combinación de parámetros en el path especificado, este va a ser el mismo que use para leer los datos
path = "C:/Users/Propietario/Desktop/ib/5-Maestría/GenData-PCA-UMAP/Datos/DatosCSV"

# GenCSVData(N, time_sample_lenght, l0, lf, tf, lcms, σs, path)

#------------------------------------------------------------------------------------------

# Se pueden leer los datos generados y guardarlos de manera mas comprimida quendandose solo con las señales y las distribuciones

# Lectura de los datos que se generaron, devuelve un array de 3 dimensiones con los datos de las señales y las distribuciones especificado cada lcm y σ

Probabilitys, Signals = ReadCSVData(N, time_sample_lenght, l0, lf, tf, lcms, σs, path)

# Para esto tenemos que hacer un reshape de los datos para que queden en un arreglo de 2 dimensiones

length_σs =  length(σs)
length_lcms = length(lcms)
length_t = length(t)
max_length = maximum(length.([t, lc]))


function reshape_data(old_matrix, old_shape, new_shape)

    old_matrix = old_matrix

    dim1, dim2, dim3 = old_shape

    new_matrix = zeros(Float64, dim3, dim1*dim2)

    for i in 1:dim1
        for j in 1:dim2
            for k in 1:dim3
                new_matrix[k,(i - 1) * dim2 + j] = old_matrix[i,j,k]
            end
        end
    end

    return new_matrix

end

# Nuevo tamaño de los datos
new_size = (length_σs * length_lcms, max_length)

# Ahora si tenemos los datos de entrada y salida es decir las señales y las distribuciones de probabilidad
dataSignals = reshape_data(Signals, size(Signals), new_size)
dataProbd = reshape_data(Probabilitys, size(Probabilitys), new_size)

# En un momento para tener un DataFrame llenamos los datos de la señal con 0s los sacamos de cada columna.
dataSignals = dataSignals[1:length_t, :]

# Ahora podemos guardar estos datos para hacer un pre procesamiento antes de utilizarlos como entrada de una red neuronal

# Elegir el path donde se van a guardar los datos
path_save = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_PCA"

df_dataSignals = DataFrame(dataSignals, :auto)
df_dataProbd = DataFrame(dataProbd, :auto)

CSV.write(path_save * "\\dataSignals.csv", df_dataSignals)
CSV.write(path_save * "\\dataProbd.csv", df_dataProbd)

