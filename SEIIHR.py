#Autor: Juan Diego Mejía Becerra
#Correo: judmejiabe@unal.edu.co

#Librerías
import datetime
import numpy as np
import plotly.graph_objs as go
import pandas as pd

from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from scipy.integrate import odeint
from scipy.interpolate import interp1d

#Las variables de este código tienden a mantener la misma notación del 
#documento relacionado.
class SEIIHR(object):
    """Esta clase contiene los métodos y funciones utilizados
    necesarios para realizar simulaciónes con el modelo SEIIHR."""
    
    secondsPerDay = 86400
    
    def __init__(self, 
                 deltaM,
                 deltaHR,
                 deltaHD,
                 deltaUR,
                 omega,
                 sigmaC,
                 sigmaHD,
                 sigmaUD,
                 gammaM,
                 gammaHR,
                 gammaR,
                 nu):
        
        #Probabilidad de infección suave
        self.deltaM = deltaM
        
        #Probabilidad de requerir hospitalización y recuperarse
        self.deltaHR = deltaHR
        
        #Probabilidad de requerir hospitalización y fallecer
        self.deltaHD = deltaHD
        
        #Probabilidad de requerir UCI y recuperarse
        self.deltaUR = deltaUR
        
        #Inverso del tiempo promedio de latencia
        self.omega = omega
        
        #Inverso del tiempo promedio pre-hospitalizado
        self.sigmaC = sigmaC
        
        #Inverso del tiempo promedio en hospitalización general antes de 
        #fallecer
        self.sigmaHD = sigmaHD
        
        #Inverso del tiempo promedio en UCI antes de fallecer
        self.sigmaUD = sigmaUD
        
        #Inverso del tiempo promedio de recuperación para una infección
        #leve
        self.gammaM = gammaM
        
        #Inverso del tiempo promedio en hospitalización general antes
        #de recuperarse
        self.gammaHR = gammaHR
        
        #Inverso del tiempo promedio en cama de recuperación
        self.gammaR = gammaR
        
        #Inverso del tiempo promedio antes de pasar a cama de
        #recuperación
        self.nu = nu
        
    def set_transmisibility_functions(self, 
                                      betaM,
                                      betaC,
                                      betaHR,
                                      betaUR,
                                      betaHD,
                                      betaUD,
                                      betaR):
        """Este método sirve para declarar las funciones que se
        utilizarán como tasas de transmisión"""
        
        self.betaM = betaM
        self.betaC = betaC
        self.betaHR = betaHR
        self.betaUR = betaUR
        self.betaHD = betaHD
        self.betaUD = betaUD
        self.betaR = betaR
        
    

            
    
    def differential_equation(self, y, t):
        """Define el sistema de ecuaciones diferenciales a resolver"""
        
        #Alias de las variables para facilitar escritura
        betaM = self.betaM
        betaC = self.betaC
        betaHR = self.betaHR
        betaUR = self.betaUR
        betaHD = self.betaHD
        betaUD = self.betaUD
        betaR = self.betaR
        deltaM = self.deltaM
        deltaHR = self.deltaHR
        deltaHD = self.deltaHD
        deltaUR = self.deltaUR
        omega = self.omega
        sigmaC = self.sigmaC
        sigmaHD = self.sigmaHD
        sigmaUD = self.sigmaUD
        gammaM = self.gammaM
        gammaHR = self.gammaHR
        gammaR = self.gammaR
        nu = self.nu
        
        
        #Valores previos de las cantidades
        S, E, IM, IC, IHR, IUR, IHD, IUD, IR, R, D, N, C, H, U = y
        
        #Sistema de ecuaciones        
        dSdt = - S * (betaM(t) * IM + betaC(t) * IC + betaHR(t) * IHR + betaUR(t) * IUR + \
                      betaR(t) * IR + betaHD(t) * IHD + betaUD(t) * IUD) / N
        
        dEdt = S * (betaM(t) * IM + betaC(t) * IC + betaHR(t) * IHR + betaUR(t) * IUR + \
                      betaR(t) * IR + betaHD(t) * IHD + betaUD(t) * IUD) / N - omega * E
        
        dIMdt = deltaM * omega * E - gammaM * IM
        
        dICdt = (1 - deltaM) * omega * E - sigmaC * IC
        
        dIHRdt = deltaHR * sigmaC * IC - gammaHR * IHR
        
        dIURdt = deltaUR * sigmaC * IC - nu * IUR
        
        dIHDdt = deltaHD * sigmaC * IC - sigmaHD * IHD
        
        dIUDdt =  (1 - deltaHR - deltaUR - deltaHD) * sigmaC * IC - sigmaUD * IUD
        
        dIRdt = nu * IUR - gammaR * IR
        
        dRdt = gammaM * IM + gammaHR * IHR + gammaR * IR
        
        dDdt = sigmaHD * IHD + sigmaUD * IUD
        
        dNdt = - (sigmaHD * IHD + sigmaUD * IUD)
        
        #Acumulado Infectados
        dCdt = S * (betaM(t) * IM + betaC(t) * IC + betaHR(t) * IHR + betaUR(t) * IUR + \
                      betaR(t) * IR + betaHD(t) * IHD + betaUD(t) * IUD) / N
        
        #Acumulado hospitalización general
        dHdt = (deltaHR + deltaHD) * sigmaC * IC
        
        #Acumulado UCI
        dUdt = (1 - deltaHR - deltaHD) * sigmaC * IC
        
        return(dSdt, dEdt, dIMdt, dICdt, dIHRdt, dIURdt, dIHDdt, dIUDdt, dIRdt, dRdt, dDdt, dNdt, dCdt, dHdt, dUdt)
    
    
    def solve_SEIIHR(self, E_0, IM_0, IC_0, IHR_0, IUR_0, IHD_0, IUD_0, IR_0, \
                     R_0, D_0, N_0, C_0, H_0, U_0, T, initialDate):
        """Resuelve las ecuaciones diferenciales y calcula el número final
        de individuos retirados. R_0 en este caso es la proporción inicial 
        de individuos retirados. T nos dá la ventana máxima de tiempo 
        donde se quiere observar el fenómeno"""
        
        #Declara los valores iniciales dentro de la instancia del objeto
        self.E_0 = E_0
        self.IM_0 = IM_0
        self.IC_0 = IC_0
        self.IHR_0 = IHR_0
        self.IUR_0 = IUR_0
        self.IHD_0 = IHD_0
        self.IUD_0 = IUD_0
        self.IR_0 = IR_0
        self.R_0 = R_0
        self.D_0 = D_0
        self.N_0 = N_0
        self.C_0 = C_0
        self.H_0 = H_0
        self.U_0 = U_0
        
        #Tiempo de simulación
        self.T = T
        
        #Fecha inicial de simulación
        self.initialDate = initialDate
        
        #Crea una partición de tamaño 30.000 del intervalo [0, T]
        self.t = np.linspace(0., T, 30000)
        
        #Traduce la grilla anterior a fechas
        self.timestamps = initialDate.timestamp() + self.t * self.secondsPerDay
        self.tDate = np.array([datetime.datetime.fromtimestamp(ts) for ts in self.timestamps])
        
        #Define la cantidad inicial de susceptibles y declara este valor en la instancia de la clase
        S_0 = N_0 - E_0 - IM_0 - IC_0 - IHR_0 - IUR_0 - IHD_0 - IUD_0 - IR_0 - R_0 - D_0
        self.S_0 = S_0
        
        #Genera un vector de valores iniciales
        self.y0 = (S_0, E_0, IM_0, IC_0, IHR_0, IUR_0, IHD_0, IUD_0, IR_0, R_0, D_0, N_0, C_0, H_0, U_0)
        
        #Solución numérica de las ecuaciones diferenciales
        ret = odeint(func = self.differential_equation, y0 = self.y0, t = self.t)
        
        #Extrae la solución numérica de las ecuaciones
        self.S, self.E, self.IM, self.IC, self.IHR, self.IUR, self.IHD,\
        self.IUD, self.IR, self.R, self.D, self.N, self.C, self.H, self.U = ret.T
        
        
        #Hospitalización general
        self.HG = self.IHR + self.IHD + self.IR
        
        #Demanda UCI
        self.UCI = self.IUR + self.IUD 
        
        #Total Infectados
        self.I = self.E + self.IM + self.IC + self.HG + self.UCI 
        
        #Genera una interpolación de la solución numerica de las ecuaciones
        self.S_ = interp1d(self.t, self.S)
        self.E_ = interp1d(self.t, self.E)
        self.IM_ = interp1d(self.t, self.IM)
        self.IC_ = interp1d(self.t, self.IC)
        self.IHR_ = interp1d(self.t, self.IHR)
        self.IUR_ = interp1d(self.t, self.IUR)
        self.IHD_ = interp1d(self.t, self.IHD)
        self.IUD_ = interp1d(self.t, self.IUD)
        self.IR_ = interp1d(self.t, self.IR)
        self.R_ = interp1d(self.t, self.R)
        self.D_ = interp1d(self.t, self.D)
        self.N_ = interp1d(self.t, self.N)
        self.C_ = interp1d(self.t, self.C)
        self.H_ = interp1d(self.t, self.H)
        self.U_ = interp1d(self.t, self.U)
        self.HG_ = interp1d(self.t, self.HG)
        self.UCI_ = interp1d(self.t, self.UCI)
        self.I_ = interp1d(self.t, self.I)
        
    def Rt_scalar(self, t):
        """Calcula el número efectivo de reproducción para el escalar t"""
        betaM = self.betaM
        betaC = self.betaC
        betaHR = self.betaHR
        betaUR = self.betaUR
        betaHD = self.betaHD
        betaUD = self.betaUD
        betaR = self.betaR
        deltaM = self.deltaM
        deltaHR = self.deltaHR
        deltaHD = self.deltaHD
        deltaUR = self.deltaUR
        omega = self.omega
        sigmaC = self.sigmaC
        sigmaHD = self.sigmaHD
        sigmaUD = self.sigmaUD
        gammaM = self.gammaM
        gammaHR = self.gammaHR
        gammaR = self.gammaR
        nu = self.nu
        
        
        self.Tr = np.matrix([[0, betaM(t), betaC(t), betaHR(t), betaUR(t), betaHD(t), betaUD(t), betaR(t)]])
        
        
        
        self.Sigma = np.matrix([[- omega, 0, 0, 0, 0, 0, 0, 0],
                                [deltaM * omega, - gammaM, 0, 0, 0, 0, 0, 0],
                                [(1 - deltaM) * omega, 0, - sigmaC, 0, 0, 0, 0, 0],
                                [0, 0, deltaHR * sigmaC, - gammaHR, 0, 0, 0, 0],
                                [0, 0, deltaUR * sigmaC, 0, - nu, 0, 0, 0],
                                [0, 0, deltaHD * sigmaC, 0, 0, - sigmaHD, 0, 0],
                                [0, 0, (1 - deltaHR - deltaUR - deltaHD) * sigmaC, 0, 0, 0, - sigmaUD, 0],
                                [0, 0, 0, 0, nu, 0, 0, - gammaR]])
        
        #print(np.matmul(self.T, np.linalg.inv(self.Sigma)))
        
        
        return(- np.matmul(self.Tr, np.linalg.inv(self.Sigma))[0, 0] * self.S_(t) / self.N_(t))
        
    
    def Rt_(self, t):
        """Calcula el número efectivo de reproducción para el vector t"""
        RT = []
        for t_ in self.t:
            RT.append(self.Rt_scalar(t_))
        RT = np.array(RT)
        return(interp1d(self.t, RT)(t))
    
    def table_to_export(self):
        """Genera un resumen diario de los resultados"""
        
        seq = np.arange(0., np.floor(self.T))
        seqTimestamps = self.initialDate.timestamp() + seq * self.secondsPerDay
        seqDates = np.array([datetime.datetime.fromtimestamp(ts) for ts in seqTimestamps])

        results = pd.DataFrame({'Día' : np.int_(seq),
                                'Fecha' : seqDates,
                               'Susceptibles' : self.S_(seq),
                               'Infectados' : self.I_(seq),
                               'Síntomas Leves y Moderados' : self.IM_(seq),
                               'Requerirán Hospitalización' : self.IC_(seq),
                               'Requieren Hospitaliación General' : self.HG_(seq),
                               'Requieren UCI' : self.UCI_(seq),
                               'Fallecidos' : self.D_(seq),
                               'Recuperados' : self.R_(seq),
                               'Acumulado Infectados' : self.C_(seq),
                               'Acumulado Hospitalizacion General' : self.H_(seq),
                               'Acumulado UCI' : self.U_(seq),
                               'Número Efectivo de Reproducción' : self.Rt_(seq + 0.1)})

        return(results)
    
    def export_table(self, fileName, scenarioName):
        """Exporta una tabla en formato excel del resumen generado con table_to_export"""
        
        table = self.table_to_export()
        table['Fecha'] = table['Fecha'].dt.strftime('%d/%m/%Y')

        
        writer = pd.ExcelWriter(fileName, engine = 'xlsxwriter')
        table.to_excel(writer, sheet_name = scenarioName, index = False)
        
        workbook = writer.book
        worksheet = writer.sheets[scenarioName]
        
        format1 = workbook.add_format({'num_format' : '#,##0.00', 'align' : 'center'})
        format2 = workbook.add_format({'align' : 'center'})
                                      
        worksheet.set_column('A:A', 10, format2)
        worksheet.set_column('B:B', 30, format2)
        worksheet.set_column('C:N', 30, format1)
        
       
                                      
        writer.save()
        
    def plot_SEIIHR(self, separated = True):
        """Grafica los resultados. Si separated == True 
        entonces genera gráficas independientes"""
        
        
        # Create traces
        
        w = 1.2
        
        
        traceS = go.Scatter(
                x = self.tDate,
                y = self.S,
                mode = 'lines',
                name = 'Susceptibles',
                line = dict(color = 'blue',
                            width = w)
                )
        
        traceE = go.Scatter(
                x = self.tDate,
                y = self.E,
                mode = 'lines',
                name = 'Expuestos',
                line = dict(color = 'pink',
                            width = w)
                )
        
        traceIM = go.Scatter(
                x = self.tDate,
                y = self.IM,
                mode = 'lines',
                name = 'Leves y Moderados',
                line = dict(color = 'cyan',
                            width = w)
                )
        
        
        traceHG = go.Scatter(
                x = self.tDate,
                y = self.HG,
                mode = 'lines',
                name = 'Hospitalización General',
                line = dict(color = 'darkblue',
                            width = w)
                )

        
        traceUCI = go.Scatter(
                x = self.tDate,
                y = self.UCI,
                mode = 'lines',
                name = 'Críticos',
                line = dict(color = 'red',
                            width = w)
                )
        
        traceR = go.Scatter(
                x = self.tDate,
                y = self.R,
                mode = 'lines',
                name = 'Recuperados',
                line = dict(color = 'green',
                            width = w)
                )
        
        traceD = go.Scatter(
                x = self.tDate,
                y = self.D,
                mode = 'lines',
                name = 'Fallecidos',
                line = dict(color = 'black',
                            width = w)
                )
        
        traceI = go.Scatter(
                x = self.tDate,
                y = self.I,
                mode = 'lines',
                name = 'Infectados',
                line = dict(color = 'yellow',
                            width = w)
                )
        
        traceH = go.Scatter(
                x = self.tDate,
                y = self.H,
                mode = 'lines',
                name = 'Acumulado Demanda Hospitalización General',
                line = dict(color = 'crimson',
                            width = w)
                )
        
        traceU = go.Scatter(
                x = self.tDate,
                y = self.U,
                mode = 'lines',
                name = 'Acumulado Demanda UCI',
                line = dict(color = 'darkgoldenrod',
                            width = w)
                )
        
        

        
        if separated:
        
            fig = make_subplots(rows=5, cols=2,
            subplot_titles=('Susceptibles', 
                            'Expuestos (Estado Latente)', 
                            'Leves y Moderados', 
                            'Hospitalización General',
                            'Críticos',
                            'Recuperados',
                            'Fallecidos',
                            'Infectados',
                            'Acumulado Demanda Hospitalización General',
                            'Acumulado Demanda UCI'))
        
            fig.add_trace(traceS, row = 1, col = 1)
            fig.add_trace(traceE, row = 1, col = 2)
            fig.add_trace(traceIM, row = 2, col = 1)
            fig.add_trace(traceHG, row = 2, col = 2)
            fig.add_trace(traceUCI, row = 3, col = 1)
            fig.add_trace(traceR, row = 3, col = 2)
            fig.add_trace(traceD, row = 4, col = 1)
            fig.add_trace(traceI, row = 4, col = 2)
            fig.add_trace(traceH, row = 5, col = 1)
            fig.add_trace(traceU, row = 5, col = 2)
            fig.update_layout(title = 'Proyecciones Modelo SEIIHR')
            
        else:
            fig = go.Figure()
            fig.add_trace(traceHG)
            fig.add_trace(traceUCI)
            fig.add_trace(traceD)

            
            fig.update_layout(
                title="Proyecciones Modelo SEIIHR",
                xaxis_title = "Días",
                yaxis_title = "Casos",
            )
        
        fig.layout.template = 'seaborn'
        fig.show()
