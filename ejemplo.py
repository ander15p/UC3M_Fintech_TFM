
import swaption_data as d
import numpy as np
import matplotlib.pyplot as plt
'''
Ejemplo de uso de algunas de las funciones y clases del script swaption_data

IRS cancelable
Fijo 1.5%
Pagos anuales
Vencimiento 15 años

Fechas de bermuda cada año hasta el año 10, incluido
Entrenamiento de las redes neuronales con 1000 paths y 100 EPOCHS

Para sigma = 0.01 y kapa = 0.01,
Plot del valor por montecarlo sobre las redes entrenadas y plot del valor teórico
'''
irs = d.IRS(
        t0 = 0,
        tenor = 15.,
        coupon = 0.015,
        dt_float = 1.,
        dt_coupon = 1.)


t_bermuda = [1,2,3,4,5,6,7,8,9,10]

swaption = d.SwaptionModel(t_bermuda=t_bermuda, irs=irs)

paths = 1000
swaption.train(paths * 2,
            rate_interval = (-0.01,0.04), 
            sigma_interval = (0.0005,0.02), 
            kapa_interval = (-0.05,0.05),
            model_type = 'DNN',
            verbose=True,
            val = 0.5,
            cost_derivs = 1.,
            EPOCHS = 100,
            BATCH_SIZE = 25,
            patience = 100)



rates = np.linspace(-0.01,0.04,20)
kapa = 0.01
sigma = 0.01

npv_th = []
npv_montecarlo_dnn = []

for i in range(len(rates)):
    npv_montecarlo_dnn.append(swaption.npv_sim(1000, rates[i], sigma, kapa))
    npv_th.append(d.irs_cancelable_npv_th(t_bermuda = t_bermuda,rate = rates[i],sigma = sigma,kapa=kapa))
    print(i)
    
plt.style.use('seaborn')
fontsize=12
plt.scatter(rates,npv_montecarlo_dnn,c='b',label='DNN Montecarlo')
plt.plot(rates,npv_th,c='r',label='Valor teórico')
plt.ylabel('NPV',fontsize=fontsize)
plt.xlabel('Rate',fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.show()