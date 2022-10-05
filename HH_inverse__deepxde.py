import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from functools import partial
from HodgkinHuxley import HodgkinHuxley
import deepxde as dde
import numpy as np
import torch



# device = torch.device("mps" if torch.device("mps") else "cpu") # 单GPU或者CPU
device = "cpu"# 单GPU或者CPU
print(device)

class Switch():
    def __init__(self,g_Na=120, g_K=36, g_L=0.3, E_Na=50, E_K=-77, E_L=-54.387,Is=20,Is_type='hengding'):
        self.Is = Is 
        self.Is_type = Is_type
          
        self.t    = np.arange(0, t_n, 0.01)
        self.g_Na = g_Na                             
        """ Sodium (Na) maximum conductances, in mS/cm^2 """
        
        self.g_K  = g_K                              
        """ Postassium (K) maximum conductances, in mS/cm^2 """
        
        self.g_L  = g_L                              
        """ Leak maximum conductances, in mS/cm^2 """
        
        self.E_Na = E_Na                             
        """ Sodium (Na) Nernst reversal potentials, in mV """
        
        self.E_K  = E_K                              
        """ Postassium (K) Nernst reversal potentials, in mV """
        
        self.E_L  = E_L                              
        """ Leak Nernst reversal potentials, in mV """
        
    alpha_n = lambda self, V: 0.0077514 * V + 0.57038216
    alpha_m = lambda self, V,c1,c2: c1 * V + c2
    alpha_h = lambda self, V: -0.00102785 * V + 0.00936024
    
    beta_n = lambda self, V: -0.00100503 * V + 0.06064837
    beta_m = lambda self, V,c3,c4: c3 * V + c4
    beta_h = lambda self, V: 0.01098231 * V + 0.78694058
    
    # def I_Na(self, V, m, h):
    #     return self.g_Na * m**3 * h * (V - self.E_Na)
    # def I_K(self, V, n):
    #     return self.g_K  * n**4 * (V - self.E_K)
    # def I_L(self, V):
    #     return self.g_L * (V - self.E_L)

    def I_inj(self):
        t=self.t
        # 方波刺激
        if self.Is_type=="pulsing_step":
            return self.Is*(t>10) - self.Is*(t>20) + self.Is*(t>30) -self.Is *(t>40)+ self.Is*(t>50) -self.Is *(t>60)+ self.Is*(t>70) -self.Is *(t>80)+ self.Is*(t>90) -self.Is *(t>100)

        if self.Is_type=="long_step":
            return self.Is * (t>10) - self.Is * (t>80)
        if self.Is_type=="sin":       
            return self.Is+self.Is*np.sin(t/2)
        if self.Is_type=="hengding":       
            return self.Is*(t>0)

t_n=50

runner = HodgkinHuxley(t_n=t_n,g_Na=120, g_K=36, g_L=0.3,delta_t=0.1,Is=20,Is_type="pulsing_step")
observe_t,ob_y ,i_inj= runner.Main(plot=False)

V = ob_y[:,0]
m = ob_y[:,1]
h = ob_y[:,2]
n = ob_y[:,3]

sw=Switch(Is=20,Is_type="pulsing_step")

C1 = dde.Variable(1.0) # g_Na
C2 = dde.Variable(1.0) # g_K
C3 = dde.Variable(1.0) # g_L
C4 = dde.Variable(1.0) # g_L



# Most backends
def HH(t, y):
       
    V, m, h, n= y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
    dy1_x = dde.grad.jacobian(y, t, i=0)
    dy2_x = dde.grad.jacobian(y, t, i=1)
    dy3_x = dde.grad.jacobian(y, t, i=2)
    dy4_x = dde.grad.jacobian(y, t, i=3)
    
    I_Na = lambda V, m, h: sw.g_Na * m**3 * h * (V - sw.E_Na)
    I_K = lambda V, n : sw.g_K  * n**4 * (V - sw.E_K)
    I_L = lambda V: sw.g_L * (V - sw.E_L)
        
    
    # return [
    #     dy1_x - torch.from_numpy(sw.I_inj()).to(device) + (I_Na(V,m,h)) + I_K(V,n) + I_L(V) ,
    #     dy2_x - sw.alpha_m(V,C1,C2)*(1.0-m) + sw.beta_m(V,C3)*m,
    #     dy3_x - sw.alpha_h(V)*(1.0-h) + sw.beta_h(V)*h,
    #     dy4_x - sw.alpha_n(V)*(1.0-n) + sw.beta_n(V)*n,
    # ]
    return [
        dy1_x - torch.from_numpy(sw.I_inj()) + I_Na(V,m,h) + I_K(V,n) + I_L(V) ,
        dy2_x - sw.alpha_m(V,C1,C2)*(1.0-m) + sw.beta_m(V,C3,C4)*m,
        dy3_x - sw.alpha_h(V)*(1.0-h) + sw.beta_h(V)*h,
        dy4_x - sw.alpha_n(V)*(1.0-n) + sw.beta_n(V)*n,
    ]
def boundary(_, on_initial):
    return on_initial

geom = dde.geometry.TimeDomain(0, t_n)

# Initial conditions -65, 0.05, 0.6, 0.32
ic1 = dde.icbc.IC(geom, lambda X: -65, boundary, component=0) # 数据的初始数值
ic2 = dde.icbc.IC(geom, lambda X: .05, boundary, component=1)
ic3 = dde.icbc.IC(geom, lambda X: .6, boundary, component=2)
ic4 = dde.icbc.IC(geom, lambda X: .32, boundary, component=3)

observe_t = np.expand_dims(observe_t,axis=1) # 扩充纬度
observe_y0 = dde.icbc.PointSetBC(observe_t, ob_y[:, 0:1], component=0)
observe_y1 = dde.icbc.PointSetBC(observe_t, ob_y[:, 1:2], component=1)
observe_y2 = dde.icbc.PointSetBC(observe_t, ob_y[:, 2:3], component=2)
observe_y3 = dde.icbc.PointSetBC(observe_t, ob_y[:, 3:4], component=3)

data = dde.data.PDE(
    geom,
    HH,
    [ic1, ic2,ic3,ic4, observe_y0, observe_y1, observe_y2, observe_y3],
    num_domain=600,
    num_boundary=2,
    anchors=observe_t,
    # num_test=200
)
net = dde.nn.FNN([1] + [128] * 3 + [4], "sin", "Glorot uniform")
model = dde.Model(data, net)
epochs = 60000
vari_save_filename="hh_variables_datasize_{}_epochs_{}.dat".format(ob_y.shape[0],epochs)

model.compile("adam", lr=1e-5, external_trainable_variables=[C1, C2, C3,C4])
variable = dde.callbacks.VariableValue(
    [C1, C2, C3,C4], filename=vari_save_filename
)
losshistory, train_state = model.train(iterations=epochs, callbacks=[variable])
print(variable.get_value())

# dde.saveplot(losshistory, train_state, issave=True, isplot=True)
