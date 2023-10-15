# Generación de datos de RMN Hahn
using CSV

# Poner una semilla random para reproducibilidad (opcional)
# Random.seed!(0)

# Constantes útiles
γ = 2.675e8  # Factor girómagetico del esín nuclear del proton (s⁻¹T⁻¹) de https://physics.nist.gov/cgi-bin/cuu/Value?gammap
G = 8.73e-7  # Gradiente externo (T/μm) de Validating NOGSE’s size distribution predictions in yeast cells Paper 1
D0 = 1e3 # Coeficiente de difusión (μm²/s) del Ejercico
# Voy a dejar todo en μm, s y T

#------------------------------------------------------------------------------------------

# Generación de distribution log-normal, con parámetros lcm y σ, en este caso lcm es el tamaño medio del compartimiento y σ es la desviación estándar del tamaño del compartimiento
# Mientras que σ es el ancho de la distribution de probabilidad con respecto a su media.
# Hagamos una función P(lc) de la distribucion log-normal

function P(lc,lcm,σ)
    return ( exp( -(log(lc) - log(lcm))^2 / (2σ^2) ) ) / (lc*σ*sqrt(2π))
end

#------------------------------------------------------------------------------------------

# Función M_l Magnetización de Hahn, para un tiempo t y un tamaño lc
function Ml_Hahn(t, lc)
    τc = lc^2 / (2 * D0)
    term1 = -γ^2 * G^2 * D0 * τc^2
    term2 = t - τc * (3 + exp(-t / τc) - 4 * exp(-t / (2 * τc)))
    return exp(term1 * term2)
end

#------------------------------------------------------------------------------------------

# Función S_hanh(t)
function S_han(lcm, σ, N, l0, lf, t)
    lc = range(l0, lf, length = N) # Generamos los tamaños de compartimientos lc desde l0 hasta lf

    P_lc = P.(lc,lcm, σ) # Consideramos media lcm y ancho σ
    M_lc = Ml_Hahn.(t, lc) # Calculamos M_lc(t) para cada tamaño de compartimiento
    
    S = sum(M_lc .* P_lc)
    return S
end

#------------------------------------------------------------------------------------------

# Generación de Señal de Hahn y distribución de tamaños de compartimientos

# N: cantidad de compartimientos lc
# lcm: tamaño medio de compartimiento
# σ: desviación estándar de compartimiento
# l0: tamaño mínimo de compartimiento
# lf: tamaño máximo de compartimiento
# time_sim: tiempo máximo de simulación
# time_sample_lenght: cantidad de puntos de tiempo


function GenData(N, lcm, σ, l0, lf, time_sim, time_sample_lenght)
    # Generamos los tiempos de difusión y de tamaños de compartimientos
    t = range(0, time_sim, length = time_sample_lenght)
    lc = range(l0, lf, length = N)

    # Generamos las distribuciones
    P_l = P.(lc, lcm, σ)

    # Calculamos la señal
    S0 = S_han(lcm, σ, N, l0, lf, 0)
    S = S_han.(lcm, σ, N, l0, lf, t) ./ S0
    return t, lc, S, P_l
end

#------------------------------------------------------------------------------------------

# Generación de datos en CSV
# N numero de compartimientos
# time_sample_lenght cantidad de puntos de tiempo
# l0 tamaño mínimo de compartimiento
# lf tamaño máximo de compartimiento
# tf tiempo máximo de simulación
# lcms vector de tamaños medios de compartimientos
# σs vector de desviaciones estándar

function GenCSVData(N, time_sample_lenght, l0, lf, tf, lcms, σs, path)

    function fill_missing(value, column, max_lenght)
        if length(column) < max_lenght
            return vcat(column,fill(value, max_lenght - length(column)))
        else
            return column
        end
    end


    for lcm in lcms
        for σ in σs
            t, l, S, P_l = GenData(N, lcm, σ, l0, lf, tf, time_sample_lenght)
            max_lenght = maximum(length.([l,t]))
            
            t = fill_missing(0, t, max_lenght)
            l = fill_missing(0, l, max_lenght)
            S = fill_missing(0, S, max_lenght)
            P_l = fill_missing(0, P_l, max_lenght)

            df = DataFrame(t = t, l = l, S = S, P_l = P_l)
            CSV.write(path * "/$(lcm)_$(σ)l_$N.csv", df)
        end
    end
end

#------------------------------------------------------------------------------------------

# Lectura de los datos que se generaron
# mismos parámetros que GenCSVData

function ReadCSVData(N, time_sample_lenght, l0, lf, tf, lcms, σs, path)
    t = range(0, tf, length = time_sample_lenght)
    lc = range(l0, lf, length = N)
    length_t = length(t)
    length_lc = length(lc)
    max_lenght = maximum(length.([t, lc]))

    Probabilitys = zeros(length(lcms), length(σs), max_lenght)
    Signals = zeros(length(lcms), length(σs), max_lenght)

    for lcm in lcms
        # pl = plot(xlabel = L"$t$ (s)", ylabel = L"S(t)", title = L"Señal de Hahn, $l_{cm} = $"*" $(lcm)")
        # pl2 = plot(xlabel = L"$l_c$ (μm)", ylabel = L"$P(l_c)$", title = L"Distribución de tamaños $l_{cm} = $"*" $(lcm)")
        # xlims!(pl, (0,tf))
        # xlims!(pl2, (0,lf))
        for σ in σs
            df = CSV.read(path * "/$(lcm)_$(σ)l_2k.csv", DataFrame)
            # plot!(pl,df.t[1:length_t], df.S[1:length_t], label = "lcm = $(lcm), σ = $(σ)", legend = false)
            # plot!(pl2,df.l, df.P_l, label = "lcm = $(lcm), σ = $(σ)", legend = false)

            Probabilitys[findall(x -> x == lcm, lcms), findall(x -> x == σ, σs), :] = df.P_l
            Signals[findall(x -> x == lcm, lcms), findall(x -> x == σ, σs), :] = df.S

        end
        # display(pl)
        # display(pl2)
    end

    return Probabilitys, Signals

end