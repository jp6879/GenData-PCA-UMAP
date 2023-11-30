using Plots
using LaTeXStrings
using MultivariateStats
using DataFrames
using CSV
using Statistics
using Measures
using StatsPlots
#using PlotlyJS
#using Flux

#------------------------------------------------------------------------------------------
# Traemos los mismos datos de los parametros utilizados para generar los datos, deberiamos hacer una función grande en la proxima función que genere los datos donde les pasamos
# Todos estos parámetros desde otro programa, como ahora generamos pocos datos me quedo con esto

###################################################ACORDARSE DE ESTO#################################################################

# # Parámetros fijos

# # Lo que dejamos constante es el número de compartimientos, el rango de tamaños de correlación lc, el tiempo de simulación final y el muestreo de timepos
# N = 2000
# time_sample_lenght = 100

# # Rango de tamaños de compartimientos en μm
# l0 = 0.01
# lf = 15

# # Tiempo final de simulación en s
# tf = 1

# # Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
# lc = range(l0, lf, length = N)
# t = range(0, tf, length = time_sample_lenght)

# # Parametros que se varian

# # Rango de tamaños medios de correlación en μm
# lcms = 0.5:0.01:6
# σs = 0.01:0.01:1

#------------------------------------------------------------------------------------------
# # Parámetros fijos

# # Lo que dejamos constante es el número de compartimientos, el rango de tamaños de correlación lc, el tiempo de simulación final y el muestreo de timepos
# N = 1700
# time_sample_lenght = 100

# # Rango de tamaños de compartimientos en μm
# l0 = 0.05
# lf = 10

# # Tiempo final de simulación en s
# tf = 1

# # Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
# lc = range(l0, lf, length = N)
# t = range(0, tf, length = time_sample_lenght)

# # Parametros que se varian

# # Rango de tamaños medios de correlación en μm
# lcms = 0.5:0.005:6
# σs = 0.01:0.01:1
#------------------------------------------------------------------------------------------
# Parámetros fijos

# # Lo que dejamos constante es el número de compartimientos, el rango de tamaños de correlación lc, el tiempo de simulación final y el muestreo de timepos
# N = 2000
# time_sample_lenght = 100

# # Rango de tamaños de compartimientos en μm
# l0 = 0.01
# lf = 50

# # Tiempo final de simulación en s
# tf = 1

# # Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
# lc = range(l0, lf, length = N)
# t = range(0, tf, length = time_sample_lenght)

# # Parametros que se varian

# # Rango de tamaños medios de correlación en μm
# lcms = 0.5:0.01:6
# σs = 0.01:0.01:1

# Parámetros fijos

# Lo que dejamos constante es el número de compartimientos, el rango de tamaños de correlación lc, el tiempo de simulación final y el muestreo de timepos
N = 5000
time_sample_lenght = 1000

# Rango de tamaños de compartimientos en μm
l0 = 0.01
lf = 50

# Tiempo final de simulación en s
tf = 1

# Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
lc = range(l0, lf, length = N)
t_short = collect(range(0, 0.1, length = 1000))
t_long = collect(range(0.1, 1, length = 100))

# Concatenate t_short and t_long

t = vcat(t_short, t_long)

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

# path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Little_Data\\Little_Data_CSV"
# path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\DatosCSV"
# path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_PCA\\PCAXL"
# path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_PCA2"
path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData_PCA_Final\\Datos_Final\\datos_PCA"
dataSignals = GetSignals(path_read)
dataProbd = GetProbd(path_read)

# for i in 1:size(dataProbd)[2]
#     dataProbd[:,i] = dataProbd[:,i] ./ maximum(dataProbd[:,i])
# end

# Plots.plot(lc,dataProbd[:,end], label = "lcm = $(lcms[end]), σ = $(σs[end])")

#------------------------------------------------------------------------------------------
# Funcion que centra los datos

function CenterData(Non_C_Matrix)
	data_matrix = Non_C_Matrix
	col_means = mean(data_matrix, dims = 1)
	centered_data = data_matrix .- col_means
	return centered_data
end

#------------------------------------------------------------------------------------------

# PCA para ciertos datos

function PCA_Data(dataIN)

    # Primero centramos los datos
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

reduced_data_Signals, pca_model_signals = PCA_Data(dataSignals)
reduced_data_Probd, pca_model_probd = PCA_Data(dataProbd)

Plots.plot(cumsum(principalvars(pca_model_signals)) / sum(principalvars(pca_model_signals)) * 100, label = "Varianza acumulada señales", legend = :bottomright, xlabel = "Componentes principales tomadas", ylabel = "Varianza acumulada (%)", tickfontsize=11, labelfontsize=13, legendfontsize=9, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm, marker = "o")
# savefig("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Plots_PCA_Serie2\\Varianza_acumulada_Signals.pdf")

Plots.plot(cumsum(principalvars(pca_model_probd)) / sum(principalvars(pca_model_probd)) * 100, label = "Varianza acumulada distribuciones de probabilidad", legend = :bottomright, xlabel = "Componentes principales tomadas", ylabel = "Varianza acumulada (%)", tickfontsize=11, labelfontsize=13, legendfontsize=9, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm, marker = "o")
# savefig("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Plots_PCA_Serie2\\Varianza_acumulada_Probd.pdf")


#plot!(pl, legend=:best, lw=5, tickfontsize=11, labelfontsize=13, legendfontsize=9, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm)


#------------------------------------------------------------------------------------------

# Quiero ver hasta que componente hay una varianza acumulada del 98% en las distribuciones de probabilidad
pcs_vars_s = principalvars(pca_model_signals)
pcs_vars_pd = principalvars(pca_model_probd)
projIN = projection(pca_model_signals) # Proyección de los datos sobre los
projOUT = projection(pca_model_probd) # Proyección de los datos sobre los

limdim_S = 0
limdim_P = 0
for i in 1:length(pcs_vars_s)
    if sum(pcs_vars_s[1:i]) / sum(pcs_vars_s) * 100 > 99
        println("La varianza acumulada de las señales es del ", sum(pcs_vars_s[1:i]) / sum(pcs_vars_s) * 100, "% con ", i, " componentes principales")
        limdim_S = i
        break
    end
end

for i in 1:length(pcs_vars_pd)
    if sum(pcs_vars_pd[1:i]) / sum(pcs_vars_pd) * 100 > 80
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

# re_signals = reconstruct(pca_model_signals, reduced_data_Signals)
# re_probd = reconstruct(pca_model_probd, reduced_data_Probd)

# Plots.scatter(t,re_signals[:,0*100 + 1], label = "lcm = $(lcms[1]), σ = $(σs[1])", markersize = 2)
# Plots.scatter!(t,re_signals[:,0*100 + 20], label = "lcm = $(lcms[1]), σ = $(σs[20])", markersize = 2)
# Plots.scatter!(t,re_signals[:,0*100 + 100], label = "lcm = $(lcms[1]), σ = $(σs[100])", markersize = 2)
# Plots.scatter!(t,re_signals[:,(20 - 1)*100 + 1], label = "lcm = $(lcms[20]), σ = $(σs[1])", markersize = 2)
# Plots.scatter!(t,re_signals[:,(20 - 1)*100 + 100], label = "lcm = $(lcms[20]), σ = $(σs[100])", markersize = 2)

# Plots.scatter(lc,re_probd[:,(50-1)*100 + 20], label = "lcm = $(lcms[50]), σ = $(σs[20])", markersize = 0.5)
# Plots.scatter!(lc,re_probd[:,(60-1)*100 + 100], label = "lcm = $(lcms[50]), σ = $(σs[100])", markersize = 0.5)
# Plots.scatter!(lc,re_probd[:,(551 - 1)*100 + 1], label = "lcm = $(lcms[551]), σ = $(σs[1])", markersize = 0.5)
# Plots.scatter!(lc,re_probd[:,(551 - 1)*100 + 100], label = "lcm = $(lcms[551]), σ = $(σs[100])", markersize = 0.001)

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

begin
	pl = Plots.plot(title = L"Datos $S(t)$ transformados", legend=:best, xlabel="PC 1", ylabel="PC 2")
	for i in 1:(dim1)
	    reduced_1 = zeros(100)
	    reduced_2 = zeros(100)
	    for j in 1:dim2
	        reduced_1[j] = reduced_data_Signals[1, (i - 1)*dim2 + j]
	        reduced_2[j] = reduced_data_Signals[2, (i - 1)*dim2 + j]
	    end
		if(i%30 == 0 || i == 1)
		    Plots.scatter!(pl,reduced_1, reduced_2, label= L"l_{cm} = " * "$(lcms[i])", layout=(1, 1), legend=:best, tickfontsize=11, labelfontsize=13, legendfontsize=8, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm)
		end
	end

	# for i=1:2; Plots.plot!(pl,[0,projIN[i,1]], [0,projIN[i,2]], arrow=true, label=[L"V1",L"V2"][i], legend=true); end
	pl
end
savefig("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Plots_PCA_Serie2\\PCA_Signals_lcm.pdf")

begin
	plsg = Plots.plot(title = L"Datos $S(t)$ transformados", legend=:best, xlabel="PC 1", ylabel="PC 2")

	for i in 1:dim2
		reduced_1 = zeros(dim1)
		reduced_2 = zeros(dim1)
		for j in 1:dim1
			reduced_1[j] = reduced_data_Signals[1,(j - 1)*dim2 + i]
			reduced_2[j] = reduced_data_Signals[2,(j - 1)*dim2 + i]
		end
		if(i%12 == 0)
			Plots.scatter!(plsg,reduced_1, reduced_2, label=L"σ = "* "$(σs[i])", layout=(1, 1), legend=:best, tickfontsize=11, labelfontsize=13, legendfontsize=8, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm)
		end
	end
	# for i=1:2; Plots.plot!(plsg,[0,projIN[i,1]], [0,projIN[i,2]], arrow=true, label=[L"V1",L"V2"][i], legend=true); end
	plsg
end

savefig("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Plots_PCA_Serie2\\PCA_Signals_σ.pdf")

begin
	plx = Plots.scatter(reduced_data_Signals[1, :], reduced_data_Signals[2, :], title=L"Datos $S(t)$ transformados", legend=:best, xlabel="PC 1", ylabel="PC 2", label = L"$Y = P^T(S - \bar{S})$", tickfontsize=11, labelfontsize=13, legendfontsize=8, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm)
	# for i=1:2; Plots.plot!(plx,[0,projIN[i,1]], [0,projIN[i,2]], arrow=true, label=[L"V1",L"V2"][i], legend=:best); end
	plx
end

savefig("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Plots_PCA_Serie2\\PCA_Signals.pdf")



# # Guardamos la identificacion y los datos transformados en un DataFrame para graficos, se podria tambien guardarlos en CSV

df_PCA_Signals = DataFrame(
		pc1 = reduced_data_Signals[1, 1:4:end],
	    pc2 = reduced_data_Signals[2, 1:4:end],
        pc3 = reduced_data_Signals[3, 1:4:end],
	    σs = column_σs[1:4:end],
	    lcm = column_lcm[1:4:end],
	)

# df_PCA_Probd = DataFrame(
#         pc1 = reduced_data_Probd[1, :],
#         pc2 = reduced_data_Probd[2, :],
#         pc3 = reduced_data_Probd[3, :],
#         # pc4 = reduced_data_Probd[4, :],
#         # pc5 = reduced_data_Probd[5, :],
#         # pc6 = reduced_data_Probd[6, :],
#         # pc7 = reduced_data_Probd[7, :],
#         # pc8 = reduced_data_Probd[8, :],
#         # pc9 = reduced_data_Probd[9, :],
#         # pc10 = reduced_data_Probd[10, :],
#         # pc11 = reduced_data_Probd[11, :],
#         σs = column_σs,
#         lcm = column_lcm,
#     )


# plot_lcms_S = @df df_PCA_Signals StatsPlots.scatter(
#     :pc1,
#     :pc2,
#     group = :σs,
#     marker = (1,5),
#     xaxis = (title = "PC1"),
#     yaxis = (title = "PC2"),
#     xlabel = "PC1",
#     ylabel = "PC2",
#     labels = false,  # Use the modified labels
#     title = "PCA para S(t)",
# )


# plot_lcms_PD = @df df_PCA_Probd StatsPlots.scatter(
#     :pc1,
#     :pc2,
#     group = :lcm,
#     marker = (0.4,5),
#     xaxis = (title = "PC1"),
#     yaxis = (title = "PC2"),
#     xlabel = "PC1",
#     ylabel = "PC2",
#     labels = false,  # Use the modified labels
#     title = "PCA para P(lc)"
# )

# PlotlyJS.plot(
#     df_PCA_Probd, Layout(margin=attr(l=0, r=0, b=0, t=0)),
#     x=:pc1, y=:pc2, z=:pc3, color=:σs,
#     type="scatter3d", mode="markers", hoverinfo="text", hovertext=:lcm,
# )

# # Guardamos estos datos en CSV

# # path_save = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Little_Data\\Little_Data_CSV"
# # path_save = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_PCA"

# path_save = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_PCA2"

# CSV.write(path_save * "\\df_PCA_Signals.csv", df_PCA_Signals)
# CSV.write(path_save * "\\df_PCA_Probd_n.csv", df_PCA_Probd)

#------------------------------------------------------------------------------------------