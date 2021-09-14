'''
Script con las funciones del modelo LGM
'''

import tensorflow as tf


@tf.function
def discount_curve(r,t,r_cte = False):
    if r_cte == True:
        b = tf.math.exp(-r*(t-t[0]))
    else:
        b = tf.Variable([1.],dtype='double')
        for i in range(1,t.shape[0]):
            b = tf.concat([b, [tf.math.exp(-integral_trapz(r[:i+1],t[:i+1]))]],0)
    return b

@tf.function
def integral_trapz(y, x):
    dx = (x[-1] - x[0]) / (int(x.shape[0]) - 1)
    return ((y[0] + y[-1])/2 + tf.reduce_sum(y[1:-1])) * dx  


@tf.function
def rate_from_b(b,t,r_cte = False):
    if r_cte == True:
        r = -tf.math.log(b)/(t-t[0])
    return r

@tf.function()
def zeta_vector(sigma,t,sigma_cte = False):
    if sigma_cte == True:
        zeta = sigma*sigma*t
    else:
        zeta = tf.Variable([],dtype='double')
        for i in range(0,t.shape[0]):
            zeta = tf.concat([zeta, [integral_trapz(sigma[0:i+1]**2,t[:i+1])]],0)
    return zeta

@tf.function()
def libor(discount_i1,discount_i2,year_frac,spread=0):
    return (discount_i1/discount_i2 - 1)/year_frac +spread

@tf.function()
def h_factor(t,kapa):
    return (1-tf.math.exp(-kapa*t))/kapa

@tf.function()
def numeraire(B_t,t,zeta_t,kapa,x):
    h=h_factor(t,kapa)
    exp_=tf.math.exp(h*x + 0.5*h**2*zeta_t)
    return exp_/B_t

@tf.function()
def discount_factor(i_1,i_2,t_tensor,B_tensor,zeta,kapa,x):
    HT=h_factor(t_tensor[i_2],kapa)
    Ht=h_factor(t_tensor[i_1],kapa)
    exp_term=-(HT-Ht)*x -0.5*(HT**2-Ht**2)*zeta
   # exp_term=-(HT-Ht)*x -(HT**2-Ht**2)*zeta_vector[i_1]
    BT=B_tensor[:,i_2]
    Bt=B_tensor[:,i_1]
    return (BT/Bt)*tf.exp(exp_term)#,exp_term