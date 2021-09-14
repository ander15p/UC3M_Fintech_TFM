 
import matplotlib.pyplot as plt
import utils.models as utils
from utils.LGM import *
import numpy as np
import tensorflow as tf
import logging
import os
from  tensorflow.keras.metrics import mean_squared_error
import scipy.stats
import os
import pickle
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

tf.keras.backend.clear_session()



'''
Clase para definir el IRS que utiñlizaran el resto de funciones
'''
class IRS:

    def __init__(self,
                t0 = 0,
                tenor = 15.,
                coupon = 0.015,
                dt_float = 1.,
                dt_coupon = 1.):
        
        self.t0 = t0
        self.tenor = tenor
        self.coupon = coupon
        self.dt_float = dt_float
        self.dt_coupon = dt_coupon
        return
    def set_from_dict(self,params_dict):
        for key in params_dict.keys():
            self.__dict__[key] = params_dict[key]
            
        return

'''
Función para simular los datos en una bermuda
Función para generar la simulación desde la bermuda del swaption y generar los datos para entrenar la red 
en esa bermuda. Esta función se usara de forma iterativa desde la última bermuda hasta la primera.
Las funciones y clases de los regresores estan en el scipt utils/models
'''
def sim_bermuda(paths,
                t_value, #El instante donde se calcula el valor de continuación
                t_bermuda = None,#Los instantes de tiempo donde hay bermudas a partir de t_value
                model_list = None,#Lista de modelos para hacer la regressión en cada bermuda
                scaler_list = None,#lista de objetos de la calse definida scaler, tiene que haber un scaler por cada model en model list
                rate_interval = (-0.01,0.04), 
                sigma_interval = (0.0005,0.02), 
                kapa_interval = (-0.05,0.05),
                model_type = 'DNN',
                x0mult = 1,
                irs = None, #Objeto IRS con los datos del IRS
                debug = False):
    
    if irs == None:
        irs = IRS()

    if t_bermuda == None: t_bermuda = []
    if model_list == None: model_list = []
    if scaler_list == None: scaler_list = []
   
    #Si son listas vacias será como si fuese la última bermuda
    
    
    dtype='float32'
    
    t0 = irs.t0
    tenor = irs.tenor
    coupon = irs.coupon
    dt_float = irs.dt_float
    dt_coupon = irs.dt_coupon
    t_float=np.arange(t0 + dt_float,tenor + dt_float,dt_float)
    t_coupon = np.arange(t0 + dt_coupon,tenor + dt_coupon,dt_coupon)

    t = np.concatenate(([t0],t_float,t_coupon,t_bermuda))
    t = np.sort(t)
    t = np.unique(t)
    
    i_bermuda = np.where(t_value==t)[0][0]
    
    i_ex=np.array([],dtype='int')#Índices del vector t donde hay bermuda
    for i in range(len(t_bermuda)):
        i_ex = np.append(i_ex,np.where(t_bermuda[i]==t)[0][0])
    
    i_coupon=np.array([],dtype='int')#Índices del vecotor t donde hay pago fijo
    for i in range(len(t_coupon)):
        i_coupon = np.append(i_coupon,np.where(t_coupon[i]==t)[0][0])
    
    i_float=np.array([],dtype='int')#Índices del vector t donde hay pago flotante
    for i in range(len(t_float)):
        i_float = np.append(i_float,np.where(t_float[i]==t)[0][0])
    
      
    #Variables de entrada
    if isinstance(rate_interval , tuple):
        rate = np.random.uniform(rate_interval[0],rate_interval[1],paths)
    else : 
        rate = np.random.uniform(rate_interval,rate_interval,paths)
    if isinstance(sigma_interval , tuple):
        sigma = np.random.uniform(sigma_interval[0],sigma_interval[1],paths)#de momento sigma y kapa ctes
    else:
        sigma = np.random.uniform(sigma_interval,sigma_interval,paths)
    if isinstance(kapa_interval , tuple):
        kapa = np.random.uniform(kapa_interval[0],kapa_interval[1],paths)
    else:
        kapa = np.random.uniform(kapa_interval,kapa_interval,paths)
    
    with tf.GradientTape() as tape:
        
        coupon = tf.constant(coupon,dtype=dtype)
        i_ex = tf.constant(i_ex)
        i_coupon = tf.constant(i_coupon)
        i_float = tf.constant(i_float)
        t = tf.constant(t,dtype=dtype)
        
        #rate,kapa y sigma son ctes de momento
        sigma = tf.Variable(sigma,dtype=dtype)
        kapa = tf.Variable(kapa,dtype=dtype)
        rate = tf.Variable(rate,dtype=dtype)
        r = tf.fill([paths,t.shape[0]],1.) * tf.expand_dims(rate, axis=-1)
        b = discount_curve(r,t,True)
        zeta = zeta_vector(tf.expand_dims(sigma, axis=1),t,True)

        std_mult = tf.constant(x0mult,dtype=dtype) # multiplico a la volatilidad por 2 para tener más extremos
        
        #Aqui empezaria el bucle de la bermuda i
        if t_value == 0:
            x_input = tf.fill([paths],0.)
        else:
            x_input = tf.random.normal([paths],0, std_mult*(zeta[:,i_bermuda] - zeta[:,0])**0.5) #X En la fecha de cálculo
        #A partir de x_input se propagan el resto de x para obtener numerarios
        #ex_i es la fecha de ejerccicio del swaption europeo
        
        bi = b[:,i_bermuda]
        n = tf.reshape(numeraire(bi, t[i_bermuda], zeta[:,i_bermuda], kapa, x_input),(paths,1))
        x = tf.reshape(x_input,(paths,1))
        b_bermuda = tf.reshape(discount_factor(0,0,t[i_bermuda:],b[:,i_bermuda:],zeta[:,i_bermuda],kapa,x_input),(paths,1))
        #Ahora calcula las curvas de descuento de cada path
        
        for i in range(1,t.shape[0]-i_bermuda):
            #tf.print(i,i + i_bermuda,t[i + i_bermuda])
            x = tf.concat([x,tf.reshape(tf.random.normal([paths],x[:,i-1], (zeta[:,i_bermuda + i ] - zeta[:,i_bermuda + i -1])**0.5),[paths,1])],1)
            n = tf.concat([n,tf.reshape(numeraire(b[:,i_bermuda + i], t[i_bermuda + i], zeta[:,i_bermuda + i], kapa, x[:,i]),(paths,1))],1)
            b_bermuda = tf.concat([b_bermuda, tf.reshape(discount_factor(0,i,t[i_bermuda:],b[:,i_bermuda:],zeta[:,i_bermuda],kapa,x_input),(paths,1))],1)
        
        
        #Ahora con los numerarios calculo los cash flows
        #Hasta ahora era igual que el caso europeo pero ahra tengo que utilizar los
        #regresores de las siguientes bermudas
        #El objetivo es determiar la bermuda de parada de cada path, que en el caso del irs cancelable
        #será la  primera con valor de continuación menor que 0
        #Una veza calculada la posición de esa bemuda se obtienen los CF hasta esa bermuda
        
        
        if len(t_bermuda)>0:
            if model_type == 'DNN':
                x_scaled = scaler_list[0].adapt_x(tf.stack([x[:,i_ex[0]-i_bermuda],r[:,i_ex[0]],sigma,kapa],1))
                y_scaled = tf.reshape(model_list[0](x_scaled)[0],(paths,1))
                vc_sim = scaler_list[0].unscale_y(y_scaled)#Valores de continuación siguiente bermuda
                #model.predict es una functión a definir aún
                if len(t_bermuda)>1:
                    for i in range(1,len(i_ex)):
                        x_scaled = scaler_list[i].adapt_x(tf.stack([x[:,i_ex[i]-i_bermuda],r[:,i_ex[i]],sigma,kapa],1))
                        y_scaled = tf.reshape(model_list[i](x_scaled)[0],(paths,1))
                        vc_sim = tf.concat([vc_sim, scaler_list[i].unscale_y(y_scaled) ],1)
                k = 0.
                match_indices=tf.where(vc_sim < k,
                             x=tf.range(tf.shape(vc_sim)[1]) * tf.ones_like(vc_sim,'int32'),  
                             y=(tf.shape(vc_sim)[1])*tf.ones_like(vc_sim,'int32'))
                stop_vec = tf.reduce_min(match_indices, axis=1)
                i_stop = tf.gather(tf.concat([i_ex,[t.shape[0]-1]],0),stop_vec)
                t_stop = tf.gather(t,i_stop)
            if model_type == 'VanillaNN':
                x_scaled = scaler_list[0].adapt_x(tf.stack([x[:,i_ex[0]-i_bermuda],r[:,i_ex[0]],sigma,kapa],1))
                y_scaled = tf.reshape(model_list[0](x_scaled),(paths,1))
                vc_sim = scaler_list[0].unscale_y(y_scaled)#Valores de continuación siguiente bermuda
                #model.predict es una functión a definir aún
                if len(t_bermuda)>1:
                    for i in range(1,len(i_ex)):
                        x_scaled = scaler_list[i].adapt_x(tf.stack([x[:,i_ex[i]-i_bermuda],r[:,i_ex[i]],sigma,kapa],1))
                        y_scaled = tf.reshape(model_list[i](x_scaled),(paths,1))
                        vc_sim = tf.concat([vc_sim, scaler_list[i].unscale_y(y_scaled) ],1)
                k = 0.
                match_indices=tf.where(vc_sim < k,
                             x=tf.range(tf.shape(vc_sim)[1]) * tf.ones_like(vc_sim,'int32'),  
                             y=(tf.shape(vc_sim)[1])*tf.ones_like(vc_sim,'int32'))
                stop_vec = tf.reduce_min(match_indices, axis=1)
                i_stop = tf.gather(tf.concat([i_ex,[t.shape[0]-1]],0),stop_vec)
                t_stop = tf.gather(t,i_stop)        
        else: #Es la útima bermuda, no hay bermuda_list, se va hasta maturity
            t_stop = tf.fill(paths,t[-1])
            
        bool_cf = tf.fill((paths,1),True) #Supongo en t[i_bermuda]  se continua siempre
        for i in range(i_bermuda + 1,len(t)): 
            bool_cf=tf.concat([bool_cf,tf.reshape(tf.math.greater_equal(t_stop,t[i]),(-1,1))],1)                   
         
        bool_cf = tf.cast(bool_cf,dtype)
        #bool_cf tendra las mismas dimensiones que la x simulada
        #stop_vec contiene la posición de la columna de vc donde el valor de continuación es negativo
        
        #Pata fija CF y Anualidad 1 esperada(para par rate) y teórico del IRS sin bermudas
        #El par y el NPV del IRS en la bermuda no hace falta pero se puede utilizar para hacer debug de la última bermuda
        cf_coupon = tf.fill([paths],0.)
        npv_annuity = tf.fill([paths],0.)
        npv_coupon = tf.fill([paths],0.)
        i_range_coupon = tf.where(i_coupon>i_bermuda)[:,0]
        
        if i_bermuda > 0:
            j = i_coupon[i_range_coupon[0]]-i_bermuda
            cf_coupon += coupon  * (t[i_coupon[i_range_coupon[0]]] - t[i_coupon[i_range_coupon[0]-1]]) * n[:,0]/ n[:,j] * bool_cf[:,j]
        elif i_bermuda == 0:
            j = i_coupon[0]
            cf_coupon += coupon  * (t[i_coupon[0]] - t[0]) * n[:,0]/ n[:,j] * bool_cf[:,j]
        for i in i_range_coupon[1:]:
            j = i_coupon[i]-i_bermuda
            #tf.print(j)
            cf_coupon += coupon  * (t[i_coupon[i]] - t[i_coupon[i-1]]) * n[:,0]/ n[:,j] * bool_cf[:,j]
            npv_annuity += 1 * (t[i_coupon[i]] - t[i_coupon[i-1]]) * b_bermuda[:,j]
            npv_coupon += coupon * (t[i_coupon[i]] - t[i_coupon[i-1]]) * b_bermuda[:,j]
            
        #Pata flotante CF y teóricoIRS(para par rate)
        cf_float = tf.fill([paths],0.)
        npv_float = tf.fill([paths],0.)
        i_range_float = tf.where(i_float>i_bermuda)[:,0]
        if i_bermuda > 0:
            j = i_float[i_range_coupon[0]]-i_bermuda
            b1 = n[:,0]/n[:,j-1]
            b2 = n[:,0]/n[:,j]
            cf_float += libor(b1,b2,(t[i_float[i_range_float[0]]] - t[i_float[i_range_float[0]-1]])) * (t[i_float[i_range_float[0]]] - t[i_float[i_range_float[0]-1]]) * b2 *  bool_cf[:,j]
        elif i_bermuda == 0:
            j = i_coupon[0]
            b1 = n[:,0]/n[:,j-1]
            b2 = n[:,0]/n[:,j]
            cf_float += libor(b1,b2,(t[i_float[0]] - t[0])) * (t[i_float[0]] - t[0]) * b2 *  bool_cf[:,j]
        for i in i_range_float[1:]:
            j = i_float[i]-i_bermuda
            #tf.print(j)
            b1 = n[:,0]/n[:,j-1]
            b2 = n[:,0]/n[:,j]
            cf_float += libor(b1,b2,(t[i_float[i]] - t[i_float[i-1]])) * (t[i_float[i]] - t[i_float[i-1]]) * b2 *  bool_cf[:,j]
            npv_float += libor(b_bermuda[:,j-1], b_bermuda[:,j],(t[i_float[i]] - t[i_float[i-1]])) * (t[i_float[i]] - t[i_float[i-1]]) * b_bermuda[:,j]
    
    
        cf = cf_float - cf_coupon
        par_rate = npv_float/npv_annuity
        npv = npv_float - npv_coupon
    
    if t_value == 0:
        grad = tape.gradient(cf,{'rate':rate,'vol':sigma,'k':kapa})
        #labels X
        X2 =  tf.reshape(rate,[-1,1])
        X3 =  tf.reshape(sigma,[-1,1])
        X4 =  tf.reshape(kapa,[-1,1])
        X = tf.concat([X2,X3,X4],1)
        
        #Valor de continuación taget Y
        Y  = tf.reshape(cf,[-1,1])
        
        #Diferenciales target Z
        
        Z2 = tf.reshape(grad['rate'],[-1,1])
        #Z2 = tf.fill((paths,1),0.0)
        Z3 = tf.reshape(grad['vol'],[-1,1])
        Z4 = tf.reshape(grad['k'],[-1,1])
        Z =  tf.concat([Z2,Z3,Z4],1)        
        
    else:
        grad = tape.gradient(cf,{'x':x_input,'rate':rate,'vol':sigma,'k':kapa})
        
        
        #labels X
        X1 =  tf.reshape(x_input,[-1,1])
        X2 =  tf.reshape(rate,[-1,1])
        X3 =  tf.reshape(sigma,[-1,1])
        X4 =  tf.reshape(kapa,[-1,1])
        X = tf.concat([X1,X2,X3,X4],1)
        
        #Valor de continuación taget Y
        Y  = tf.reshape(cf,[-1,1])
        
        #Diferenciales target Z
        
        Z1 = tf.reshape(grad['x'],[-1,1])
        Z2 = tf.reshape(grad['rate'],[-1,1])
        #Z2 = tf.fill((paths,1),0.0)
        Z3 = tf.reshape(grad['vol'],[-1,1])
        Z4 = tf.reshape(grad['k'],[-1,1])
        Z =  tf.concat([Z1,Z2,Z3,Z4],1)
    
    if debug == True:
        return X,Y,Z,len(tf.where(t_stop<irs.tenor))/len(t_stop),b_bermuda,par_rate,npv
    else:
        return X,Y,Z

'''
Función para entrenar la red neuronal diferencial en la bermuda
val es la proporción de datos usados para validación
patience es la cantidad de epoch sin mejorar para early stopping(Si no se quiere usar elegir patience>EPOCHS)
cost_derivs es el parámetro lambda que determina el coste de los diferenciales erróneos, si es 0 es una red neuronal normal
'''
def train_DNN(X,Y,Z,val = 0.5, cost_derivs = 1.,EPOCHS = 1000, BATCH_SIZE = 20,verbose=True, patience = 200):#Entrena el regresor y devuelve model,scaler e history
    alpha = 1/(1+cost_derivs*Z.shape[1])
    if verbose: 
        EarlyStopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=patience,
                verbose=1,
                mode="min",
                baseline=None,
                restore_best_weights=True,
                )
        callbacks = [EarlyStopping,utils.PrintProgress()]
    else:
        EarlyStopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=patience,
            verbose=0,
            mode="min",
            baseline=None,
            restore_best_weights=True,
            )
        callbacks =  [EarlyStopping]
  
    i_cut = int(round(val*X.shape[0],0))
    X_val = X[0:i_cut,:]
    X_train = X[i_cut:,:]
    Z_val = Z[0:i_cut,:]
    Z_train = Z[i_cut:,:]
    Y_val = Y[0:i_cut,:]
    Y_train = Y[i_cut:,:]

    scaler = utils.basic_scaler(X_train,Y_train,Z_train) 
    x_train_scaled,y_train_scaled,z_train_scaled = scaler.adapt(X_train,Y_train,Z_train)
    x_val_scaled,y_val_scaled,z_val_scaled = scaler.adapt(X_val,Y_val,Z_val)
    
    input_dim=(X.shape[1])       
    model = utils.get_twin_net(input_dim)
    loss_derivs = utils.CustomLoss(z_train_scaled).function_derivs
    #loss_derivs = scaler.CustomLoss.function_derivs
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss={ # named losses
                'y_pred': 'mse',
                'dydx_pred' : loss_derivs #loss_derivs
            },
        #metrics = ['mse'],
        loss_weights=[alpha,1-alpha]
    )

    history = model.fit(x_train_scaled,[y_train_scaled,z_train_scaled],
        epochs=EPOCHS,
        batch_size = BATCH_SIZE,    
        #validation_split = val,
        validation_data = (x_val_scaled,[y_val_scaled,z_val_scaled]) ,
        verbose=0,
        callbacks=callbacks)
    return model, scaler, history

'''
Función para entrenar la red neuronal tradicional en la bermuda
Exactamente igual que la función trainDNN pero sin cost_derivs porque no aplica
'''
def train_vanillaNN(X,Y,Z,val = 0.5,EPOCHS = 1000, BATCH_SIZE = 20,verbose=True, patience = 200):#Entrena el regresor y devuelve model,scaler e history

    if verbose: 
        EarlyStopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=patience,
                verbose=1,
                mode="min",
                baseline=None,
                restore_best_weights=True,
                )
        callbacks = [EarlyStopping,utils.PrintProgress()]
    else:
        EarlyStopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=patience,
            verbose=0,
            mode="min",
            baseline=None,
            restore_best_weights=True,
            )
        callbacks =  [EarlyStopping]

         
    i_cut = int(round(val*X.shape[0],0))
    X_val = X[0:i_cut,:]
    X_train = X[i_cut:,:]
    Y_val = Y[0:i_cut,:]
    Y_train = Y[i_cut:,:]

    scaler = utils.basic_scaler(X_train,Y_train,0,z_true = False) 
    x_train_scaled,y_train_scaled,z_train_scaled = scaler.adapt(X_train,Y_train,0)#No necesitamos z
    x_val_scaled,y_val_scaled,z_val_scaled = scaler.adapt(X_val,Y_val,0)    

    input_dim=(X.shape[1])       
    model = utils.get_vanilla_net(input_dim)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),

        loss={ # named losses
                'y_pred': 'mse',
            },
        metrics = ['mse']
    )
 
    history = model.fit(x_train_scaled,y_train_scaled,
        epochs=EPOCHS,
        batch_size = BATCH_SIZE,    
        #validation_split = val,
        validation_data = (x_val_scaled,y_val_scaled) ,
        verbose=0,
        callbacks=callbacks)
    return model, scaler, history


def get_npv(paths,t_bermuda,model_list,scaler_list,rate,sigma,kapa,model_type='DNN',irs=None):

    X,Y,Z= sim_bermuda(paths,t_value=0,
                          t_bermuda=t_bermuda,
                          irs=irs,
                          model_list=model_list,
                          scaler_list=scaler_list,
                          rate_interval = rate,
                          sigma_interval = sigma,
                          kapa_interval = kapa,
                          model_type = model_type) 

    npv = Y.numpy().mean()
    return npv

'''
función para entreanr todas las bermudas en un instrumento IRS()
'''
def train_all(paths,
        t_bermuda = None,
        rate_interval = (-0.01,0.04), 
        sigma_interval = (0.0005,0.02), 
        kapa_interval = (-0.05,0.05),
        model_type = 'DNN',
        x0mult = 1,
        irs = None,
        verbose=True,
        val = 0.3,
        cost_derivs = 1.,
        EPOCHS = 1000,
        BATCH_SIZE = 25,
        patience = 200
        ):
    
    if irs == None:
        irs = IRS()
    
    verbose2 = False
    if verbose == 0.5:
        verbose2 = True
        verbose = False
        
    if t_bermuda == None or len(t_bermuda)==0:# No hay bermudas
        #No aplica
        pass
    elif len(t_bermuda)>0 and isinstance(t_bermuda,list):
        model_list = []
        scaler_list = []
        history_list = []
    
        #sim última bermuda
        if verbose or verbose2 : print('Entrenando en bermuda número', len(t_bermuda))
        X,Y,Z = sim_bermuda(paths,t_value = t_bermuda[-1],
                            rate_interval = rate_interval,
                            kapa_interval = kapa_interval,
                            sigma_interval = sigma_interval,
                            model_type = model_type,
                            x0mult = x0mult,
                            irs = irs)
        if model_type == 'DNN':
            model, scaler, history = train_DNN(X,Y,Z,
                                               val = val,
                                               cost_derivs = cost_derivs,
                                               EPOCHS = EPOCHS,
                                               BATCH_SIZE =BATCH_SIZE,
                                               verbose=verbose,
                                               patience=patience)
        elif model_type == 'VanillaNN':
            model, scaler, history = train_vanillaNN(X,Y,verbose=verbose)
        model_list.append(model)
        scaler_list.append(scaler)
        history_list.append(history)
    
        #ahora para el resto de bermudas
        for i in range(len(t_bermuda)-1):
            if verbose or verbose2: print('Entrenando en bermuda número', len(t_bermuda)-i-1)
            X,Y,Z = sim_bermuda(paths,t_value = t_bermuda[-2-i],
                                t_bermuda = t_bermuda[-1-i:],
                                model_list = model_list[-1-i:],
                                scaler_list = scaler_list[-1-i:],
                                rate_interval = rate_interval,
                                kapa_interval = kapa_interval,
                                sigma_interval = sigma_interval,
                                model_type = model_type,
                                x0mult = x0mult,
                                irs = irs)
            if model_type == 'DNN':
                model, scaler, history = train_DNN(X,Y,Z,
                                                   val=val,
                                                   cost_derivs = cost_derivs,
                                                   EPOCHS = EPOCHS,
                                                   BATCH_SIZE =BATCH_SIZE,
                                                   verbose=verbose,
                                                   patience=patience)
            elif model_type == 'VanillaNN':
                model, scaler, history = train_DNN(X,Y,Z,
                                                   val=val,
                                                   EPOCHS = EPOCHS,
                                                   BATCH_SIZE =BATCH_SIZE,
                                                   verbose=verbose,
                                                   patience=patience)
            model_list.insert(0, model)
            scaler_list.insert(0, scaler)
            history_list.insert(0,history)    
    
        return model_list, scaler_list, history_list

'''
Función la red en la value date de tal forma que no hace falta lanzar simulaciones.
'''
def train_t0(paths,
        t_bermuda = None,        
        model_list = None,#Lista de modelos para hacer la regressión en cada bermuda
        scaler_list = None,#lista de objetos de la calse definida scaler, tiene que haber un scaler por cada model en model list
        rate_interval = (-0.01,0.04), 
        sigma_interval = (0.0005,0.02), 
        kapa_interval = (-0.05,0.05),
        model_type = 'DNN',
        irs = None,
        val = 0.3,
        cost_derivs = 1.,
        EPOCHS = 1000,
        BATCH_SIZE = 25,
        patience = 200,
        verbose = False):
        
    if model_list[0].name =="Twin_Net":
        X,Y,Z= sim_bermuda(paths,t_value=0,
                              t_bermuda=t_bermuda,
                              irs=irs,
                              model_list=model_list,
                              scaler_list=scaler_list,
                              rate_interval =rate_interval,
                              sigma_interval = sigma_interval,
                              kapa_interval = kapa_interval,
                              model_type = "DNN")
    elif model_list[0].name =="Vanilla_Net":
        X,Y,Z= sim_bermuda(paths,t_value=0,
                              t_bermuda=t_bermuda,
                              irs=irs,
                              model_list=model_list,
                              scaler_list=scaler_list,
                              rate_interval =rate_interval,
                              sigma_interval = sigma_interval,
                              kapa_interval = kapa_interval,
                              model_type = "VanillaNN")
    if irs == None:
        irs = IRS()
        
    if t_bermuda == None or len(t_bermuda)==0:# No hay bermudas
        pass
    elif len(t_bermuda)>0 and isinstance(t_bermuda,list):
        if model_type == 'DNN':
            model, scaler, history = train_DNN(X,Y,Z,verbose=verbose,val=val,cost_derivs=cost_derivs,EPOCHS=EPOCHS,BATCH_SIZE=BATCH_SIZE,patience = patience)

        elif model_type == 'VanillaNN':
            model, scaler, history = train_vanillaNN(X,Y,verbose)        
    return model, scaler, history


def get_cv(x,rate,sigma,kapa,model,scaler):#Para obtener el CV de una bermuda
    X = tf.stack([x,rate,sigma,kapa],axis=1)
    x_scaled = scaler.adapt_x(X)
    if model.name == 'Twin_Net':
        y_scaled = model(x_scaled)[0]
    elif model.name == 'Vanilla_Net':
        y_scaled = model(x_scaled)
    y = scaler.unscale_y(y_scaled)
    return y 

def get_npv_v2(rate,sigma,kapa,model,scaler):#Para obtenr NPV usando regresion
    X = tf.stack([rate,sigma,kapa],axis=1)
    x_scaled = scaler.adapt_x(X)
    if model.name == 'Twin_Net':
        y_scaled = model(x_scaled)[0]
    elif model.name == 'Vanilla_Net':
        y_scaled = model(x_scaled)
    y = scaler.unscale_y(y_scaled)
    return y 

'''
Clase con el objeto swaption
Se inicia aportando la clase IRS y la lista con las fechas de bermuda
'''

class SwaptionModel:

    def __init__(self,t_bermuda = None ,irs = None): #Irs es el objeto irs
        if irs == None:
            self.irs = IRS()
        else:
            self.irs = irs
        self.t_bermuda = t_bermuda
        self.trained_bermudas = False
        self.trained_in_t0 = False
        
    #Para entrenar todas las redes 
    def train(self,paths,
                    trainT0 =False,
                    rate_interval = (-0.01,0.04), 
                    sigma_interval = (0.0005,0.02), 
                    kapa_interval = (-0.05,0.05),
                    model_type = 'DNN',
                    x0mult = 1,
                    verbose=True,
                    val = 0.5,
                    cost_derivs = 1.,
                    EPOCHS = 500,
                    BATCH_SIZE = 25,
                    patience = 50
                    ):
        self.params = {
                    'rate_interval':rate_interval,
                    'sigma_interval':sigma_interval,
                    'kapa_interval':kapa_interval,
                    'model_type':model_type,
                    'x_input_mult':x0mult
                    }
        model_list,scaler_list,history_list = train_all(paths,
                    t_bermuda = self.t_bermuda,
                    rate_interval = rate_interval, 
                    sigma_interval = sigma_interval, 
                    kapa_interval = kapa_interval,
                    model_type = model_type,
                    irs = self.irs,
                    verbose=verbose,
                    val = val,
                    cost_derivs = cost_derivs,
                    EPOCHS = EPOCHS,
                    BATCH_SIZE =BATCH_SIZE,
                    patience=patience)
        self.model_list = model_list
        self.scaler_list = scaler_list
        self.history_list = history_list
        if trainT0 == True:
            model,scaler,history=train_t0(paths,
                        t_bermuda = self.t_bermuda,
                        model_list = self.model_list,
                        scaler_list = self.scaler_list,
                        rate_interval = rate_interval, 
                        sigma_interval = sigma_interval, 
                        kapa_interval = kapa_interval,
                        model_type = model_type,
                        irs = self.irs,
                        verbose=verbose,
                        val = val,
                        cost_derivs = cost_derivs,
                        EPOCHS = EPOCHS,
                        BATCH_SIZE =BATCH_SIZE,
                        patience=patience)
        self.trained_bermudas = True
        
    #Para entrenar una red en t=0
    def train_t0(self,paths,
                    rate_interval = (-0.01,0.04), 
                    sigma_interval = (0.0005,0.02), 
                    kapa_interval = (-0.05,0.05),
                    model_type = 'DNN',
                    verbose=True,
                    val = 0.5,
                    cost_derivs = 1.,
                    EPOCHS = 500,
                    BATCH_SIZE = 25,
                    patience = 50
                    ):
        model,scaler,history=train_t0(paths,
                    t_bermuda = self.t_bermuda,    
                    model_list = self.model_list,
                    scaler_list = self.scaler_list,
                    rate_interval = rate_interval, 
                    sigma_interval = sigma_interval, 
                    kapa_interval = kapa_interval,
                    model_type = model_type,
                    irs = self.irs,
                    verbose=verbose,
                    val = val,
                    cost_derivs = cost_derivs,
                    EPOCHS = EPOCHS,
                    BATCH_SIZE =BATCH_SIZE,
                    patience=patience)
        self.t0_model = model
        self.t0_scaler = scaler
        self.t0_history = history
        self.trained_in_t0 = True
    
    #Para obtener NPV si se entreno una red en t = 0    
    def npv_NN(self,rate,sigma,kapa):
        return get_npv_v2(rate,sigma,kapa,self.t0_model,self.t0_scaler)
    
    #Para obtener el valor de continuación estimado por la red d euna bermuda 
    def cv_NN_bermuda(self,bermuda,x,rate,sigma,kapa):#numero de bermuda empieza en 1
        return get_cv(x,rate,sigma,kapa,self.model_list[bermuda-1],self.scaler_list[bermuda-1])
    
    #Para obtener NPV si NO se entreno una red en t = 0. Se simulan trayectorias. Más conservador.
    def npv_sim(self,paths,rate,sigma,kapa):
        return get_npv(paths,self.t_bermuda,
                    self.model_list,
                    self.scaler_list,
                    rate = rate,
                    sigma = sigma,
                    kapa = kapa,
                    model_type = self.params['model_type'],
                    irs = self.irs)
    
    #Para guardar el modelo en un directorio
    def save(self,path):
        os.mkdir(path)
        pickle.dump(self.irs,open(path + '/irs.p','wb'))
        pickle.dump(self.t_bermuda, open(path + '/t_bermuda.p','wb'))
        
        if self.trained_bermudas ==True:
            pickle.dump(self.params, open(path + '/params.p','wb'))
            
            os.mkdir(path + '/model_list_weights')
            for i in range(len(self.model_list)):
                self.model_list[i].save_weights(path + '/model_list_weights/bermuda_' + str(i) + '.h5')
            os.mkdir(path + '/scaler_list')
            for i in range(len(self.scaler_list)):
                file_scaler = open(path + '/scaler_list/bermuda_' + str(i) + '.p','wb')
                pickle.dump(self.scaler_list[i], file_scaler)
        if self.trained_in_t0 == True:
            os.mkdir(path + '/regression_t0')
            file_scaler = open(path + '/regression_t0/scaler.p','wb')
            pickle.dump(self.t0_scaler, file_scaler)
            self.t0_model.save_weights(path + '/regression_t0/model_weights.h5')

'''
Función para calcular el NPV teórico 
'''
def irs_cancelable_npv_th(
                t_bermuda = None,#Los instantes de tiempo donde hay bermudas a partir de t_value
                rate = 0.02, 
                sigma = 0.01, 
                kapa = 0.01,
                irs = None,
                debug = False):
    
    percentile = 0.999
    
    if irs == None:
        irs = IRS()

    if t_bermuda == None: t_bermuda = []
    
    z_max = scipy.stats.norm.ppf(percentile)
    z_min = -z_max
    
    t0 = irs.t0
    tenor = irs.tenor
    coupon = irs.coupon
    dt_float = irs.dt_float
    dt_coupon = irs.dt_coupon
    t_float=np.arange(t0 + dt_float,tenor + dt_float,dt_float)
    t_coupon = np.arange(t0 + dt_coupon,tenor + dt_coupon,dt_coupon)
    
    t = np.concatenate(([t0],t_float,t_coupon,t_bermuda))
    t = np.sort(t)
    t = np.unique(t)
    
    i_ex=np.array([],dtype='int')#Índices del vector t donde hay bermuda
    for i in range(len(t_bermuda)):
        i_ex = np.append(i_ex,np.where(t_bermuda[i]==t)[0][0])
    
    i_coupon=np.array([],dtype='int')#Índices del vecotor t donde hay pago fijo
    for i in range(len(t_coupon)):
        i_coupon = np.append(i_coupon,np.where(t_coupon[i]==t)[0][0])
    
    i_float=np.array([],dtype='int')#Índices del vector t donde hay pago flotante
    for i in range(len(t_float)):
        i_float = np.append(i_float,np.where(t_float[i]==t)[0][0])
        
    paths = 100
    b = discount_curve(rate,t,r_cte=True).numpy()
    b = np.tile(b,(paths,1))
    zeta = zeta_vector(sigma, t, sigma_cte = True).numpy()
    zeta = np.tile(zeta,(paths,1))
    
    x_bermudas = np.zeros((paths,len(t_bermuda)))
    n_bermudas = np.zeros((paths,len(t_bermuda)))
    cv_bermudas = np.zeros((paths,len(t_bermuda)))
    '''
    IRS NPV última bermuda
    '''
    mean = 0
    i_bermuda = np.where(t_bermuda[-1]==t)[0][0]
    std = zeta[0,i_bermuda]**0.5
    x_max = z_max*std + mean
    x_min = z_min*std + mean
    
         
    #x_input = np.random.uniform(x_min,x_max,paths) Uniform o arange¿?
    x_input = np.linspace(x_min,x_max,paths)
    
    b_bermuda = discount_factor(0,0,t[i_bermuda:],b[:,i_bermuda:],zeta[:,i_bermuda],kapa,x_input).numpy().reshape((-1,1))
    for i in range(1,len(t)-i_bermuda):
        b_bermuda = np.append(b_bermuda,
                              discount_factor(0,i,t[i_bermuda:],b[:,i_bermuda:],zeta[:,i_bermuda],kapa,x_input).numpy().reshape((-1,1)),
                              axis = 1)
        
    libor_ = libor(b_bermuda[:,0:-1],b_bermuda[:,1:],1).numpy()
    dt = 1.
    cv_last_bermuda = np.clip((libor_ * dt * b_bermuda[:,1:]  - coupon * dt *b_bermuda[:,1:]).sum(axis=1),a_min=0,a_max=None)
    n_last_bermuda = numeraire(b[:,i_ex[-1]],t[i_ex[-1]],zeta[:,i_ex[-1]],kapa,x_input).numpy()
    
    x_bermudas[:,-1] = x_input
    cv_bermudas[:,-1] = cv_last_bermuda
    n_bermudas[:,-1] = n_last_bermuda
    
    if len(t_bermuda) > 1:
    

        for k in range(0,len(t_bermuda)-1):
            i_bermuda = i_ex[-2 - k]
            i_float_bermuda = i_float[np.where(i_float == i_bermuda)[0][0] + 1 : np.where(i_float == i_ex[-1-k])[0][0]+1]
            i_coupon_bermuda = i_coupon[np.where(i_float == i_bermuda)[0][0] + 1 : np.where(i_coupon == i_ex[-1-k])[0][0]+1]
            
            mean = 0
            std = zeta[0,i_bermuda]**0.5
            x_max = z_max*std + mean
            x_min = z_min*std + mean    
            x_input = np.linspace(x_min,x_max,paths)   
            b_bermuda = discount_factor(0,0,t[i_bermuda:],b[:,i_bermuda:],zeta[:,i_bermuda],kapa,x_input).numpy().reshape((-1,1))
            for i in range(1,len(t)-i_bermuda):
                b_bermuda = np.append(b_bermuda,
                                      discount_factor(0,i,t[i_bermuda:],b[:,i_bermuda:],zeta[:,i_bermuda],kapa,x_input).numpy().reshape((-1,1)),
                                      axis = 1)
            
            n_bermuda = numeraire(b[:,i_bermuda],t[i_bermuda],zeta[:,i_bermuda],kapa,x_input).numpy()
            n_next = n_bermudas[:,-1 - k]
            x_next = x_bermudas[:,-1 - k]
            cv_next = cv_bermudas[:,-1 - k]
            cf = 0  
            for j in i_float_bermuda:
                libor_ = libor(b_bermuda[:,j - i_bermuda -1],b_bermuda[:,j - i_bermuda],t[j]-t[j-1]).numpy()
                cf += libor_ * (t[j]-t[j-1]) * b_bermuda[:,j - i_bermuda]
                
            for j in i_coupon_bermuda:
                cf -= coupon * (t[j]-t[j-1]) * b_bermuda[:,j - i_bermuda]
                
            
            std_next  = (zeta[0,i_ex[-1 - k]]-zeta[0,i_ex[-2 - k]])**0.5
            #vc en la siguiente bermuda
            cv = np.zeros_like(x_input)
            
            for j in range(len(x_input)):
                #p = np.zeros_like(x_next)
                dx =np.mean(x_next[1:] - x_next[:-1])
                p = scipy.stats.norm.cdf(x_next+dx/2,x_input[j],std_next) - scipy.stats.norm.cdf(x_next-dx/2,x_input[j],std_next)
                #for i in range(0,len(p)):
                #    p[i]=scipy.stats.norm.cdf(x_next[i]+dx/2,x_input[j],std_next) - scipy.stats.norm.cdf(x_next[i]-dx/2,x_input[j],std_next)
                integral = np.sum(p*cv_next/n_next)
                cv[j] = integral * n_bermuda[j]
        
            cv = cf + cv
            x_bermudas[:,-2 - k] = x_input
            n_bermudas[:,-2 - k] = n_bermuda
            cv_bermudas[:,-2 - k] = np.clip(cv, a_min=0, a_max=None)
    
    '''
    NPV
    '''
    #value date 

    i_float_bermuda = i_float[:np.where(i_coupon == i_ex[0])[0][0]+1]
    i_coupon_bermuda = i_coupon[:np.where(i_coupon == i_ex[0])[0][0]+1]
    
    cf = 0  
    for j in i_float_bermuda:
        libor_ = libor(b[:,j-1],b[:,j],t[j]-t[j-1]).numpy()
        cf += libor_ * (t[j]-t[j-1]) * b[:,j]
        
    for j in i_coupon_bermuda:
        cf -= coupon * (t[j]-t[j-1]) * b[:,j]
        

    cv_next = cv_bermudas[:,0]
    n_next = n_bermudas[:,0]
    x_input = np.full(paths,0.)
    cv = np.zeros_like(x_input)
    x_next = x_bermudas[:,0]
    std_next  = zeta[0,i_ex[0]]**0.5
    for j in range(len(x_input)):
        #p = np.zeros_like(x_next)
        dx =np.mean(x_next[1:] - x_next[:-1])
        p = scipy.stats.norm.cdf(x_next+dx/2,x_input[j],std_next) - scipy.stats.norm.cdf(x_next-dx/2,x_input[j],std_next)
        #for i in range(0,len(p)):
        #    p[i]=scipy.stats.norm.cdf(x_next[i]+dx/2,x_input[j],std_next) - scipy.stats.norm.cdf(x_next[i]-dx/2,x_input[j],std_next)
        integral = np.sum(p*cv_next/n_next)
        cv[j] = integral * 1


    npv = cf + cv

    if debug == True:
        return npv.mean(), cf.mean(),((libor(b[:,0:-1],b[:,1:],1.).numpy() - coupon) * b[:,1:]).sum(axis=1).mean(),b,cv_bermudas,x_bermudas,n_bermudas
    else:
        return npv.mean()
    

'''
Función para cargar un modelo guardado
'''
def load_model(path):
        t_bermuda = pickle.load(open(path + '/t_bermuda.p','rb'))
        params = pickle.load(open(path + '/params.p','rb'))
        irs = pickle.load(open(path + '/irs.p','rb'))
        n = len(t_bermuda)
        model_list = []
        scaler_list = []
        for i in range(n):
            if params['model_type'] == 'DNN':
                model = utils.get_twin_net(4)
            elif params['model_type'] =="VanillaNN":
                model = utils.get_vanilla_net(4)
            model.load_weights(path + '/model_list_weights/bermuda_' + str(i) + '.h5')
            model_list.append(model)
            scaler =  pickle.load(open(path + '/scaler_list/bermuda_' + str(i) + '.p','rb'))
            scaler_list.append(scaler)
        swaption = SwaptionModel(irs=irs,t_bermuda=t_bermuda)
        swaption.params = params
        swaption.model_list = model_list
        swaption.scaler_list = scaler_list
        return swaption  
 

