import scipy as sp
import numpy as np
import pylab as plt
from scipy.integrate import odeint

class HodgkinHuxley():
    """Full Hodgkin-Huxley Model implemented in Python"""

    """ __init__ uses optional arguments """
    """ when no argument is passed default values are used """
    
    def __init__(self, Is_type='sin',Is=10, C_m=1, g_Na=120, g_K=36, g_L=0.3, E_Na=50, E_K=-77, E_L=-54.387, t_0=0, t_n=450, delta_t=0.01, I_inj_max=0, I_inj_width=0, I_inj_trans=0):
        self.Is = Is 
        self.Is_type = Is_type
        self.C_m  = C_m                              
        """ membrane capacitance, in uF/cm^2 """
        
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
        
        self.t    = np.arange(t_0, t_n, delta_t)     
        """ The time to integrate over """
        
        """ Advanced input - injection current (single rectangular pulse only) """
        
        self.I_inj_max   = I_inj_max
        """ maximum value or amplitude of injection pulse """
        
        self.I_inj_width = I_inj_width
        """ duration or width of injection pulse """
        
        self.I_inj_trans = I_inj_trans
        """ strart time of injection pulse or tranlation about time axis """

    def alpha_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.1*(V+40.0)/(1.0 - np.exp(-(V+40.0) / 10.0))

    def beta_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 4.0*np.exp(-(V+65.0) / 18.0)

    def alpha_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.07*np.exp(-(V+65.0) / 20.0)

    def beta_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 1.0/(1.0 + np.exp(-(V+35.0) / 10.0))

    def alpha_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.01*(V+55.0)/(1.0 - np.exp(-(V+55.0) / 10.0))

    def beta_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.125*np.exp(-(V+65) / 80.0)

    def I_Na(self, V, m, h):
        """
        Membrane current (in uA/cm^2)
        Sodium (Na = element name)

        |  :param V:
        |  :param m:
        |  :param h:
        |  :return:
        """
        return self.g_Na * m**3 * h * (V - self.E_Na)

    def I_K(self, V, n):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_K  * n**4 * (V - self.E_K)
    #  Leak
    def I_L(self, V):
        """
        Membrane current (in uA/cm^2)
        Leak

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_L * (V - self.E_L)

    def I_inj(self, t):
        """
        External Current

        |  :param t: time
        |  :return: step up to 10 uA/cm^2 at t>100
        |           step down to 0 uA/cm^2 at t>200
        |           step up to 35 uA/cm^2 at t>300
        |           step down to 0 uA/cm^2 at t>400
        """
        
        """ running standalone python script """
        # if __name__ == '__main__':     
        # return 10*(t>10) - 10*(t>20) + 35*(t>30) - 35*(t>40)
        # 方波刺激
        if self.Is_type=="pulsing_step":
            return self.Is*(t>10) - self.Is*(t>20) + self.Is*(t>30) -self.Is *(t>40)+ self.Is*(t>50) -self.Is *(t>60)+ self.Is*(t>70) -self.Is *(t>80)+ self.Is*(t>90) -self.Is *(t>100)
        
        if self.Is_type=="long_step":
            return self.Is * (t>10) - self.Is * (t>80)
        if self.Is_type=="sin":       
            return self.Is+self.Is*np.sin(t/2)
        if self.Is_type=="hengding":       
            return self.Is*(t>0)


    @staticmethod
    def dALLdt(X, t, self):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
        """
        V, m, h, n = X

        dVdt = (self.I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        dmdt = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m
        dhdt = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h
        dndt = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n
        return dVdt, dmdt, dhdt, dndt

    def Main(self,plot=False):
        """
        Main demo for the Hodgkin Huxley neuron model
        """

        X = odeint(self.dALLdt, [-65, 0.05, 0.6, 0.32], self.t, args=(self,))
        V = X[:,0]
        m = X[:,1]
        h = X[:,2]
        n = X[:,3]
        ina = self.I_Na(V, m, h)
        ik = self.I_K(V, n)
        il = self.I_L(V)
        
        #increase figure and font size for display in jupyter notebook
        if __name__ != '__main__':        
            plt.rcParams['figure.figsize'] = [12, 8]
            plt.rcParams['font.size'] = 15
            plt.rcParams['legend.fontsize'] = 12
            plt.rcParams['legend.loc'] = "upper right"

        if plot==True:    
            fig=plt.figure()
            
            ax1 = plt.subplot(3,1,1)
            plt.xlim([np.min(self.t),np.max(self.t)])  #for all subplots
            plt.title('Hodgkin-Huxley Neuron')
            i_inj_values = [self.I_inj(t) for t in self.t]
            plt.plot(self.t, i_inj_values, 'k')
            plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')      

            # plt.subplot(4,1,2, sharex = ax1)
            # plt.plot(self.t, ina, 'c', label='$I_{Na}$')
            # plt.plot(self.t, ik, 'y', label='$I_{K}$')
            # plt.plot(self.t, il, 'm', label='$I_{L}$')
            # plt.ylabel('Current')
            # plt.legend()

            plt.subplot(3,1,2, sharex = ax1)
            plt.plot(self.t, m, 'r', label='m')
            plt.plot(self.t, h, 'g', label='h')
            plt.plot(self.t, n, 'b', label='n')
            plt.ylabel('Gating Value')
            plt.legend()

            plt.subplot(3,1,3, sharex = ax1)
            plt.plot(self.t, V, 'k')
            plt.ylabel('V (mV)')
            plt.xlabel('t (ms)')
            #plt.ylim(-1, 40)
            
            plt.tight_layout()
            # plt.savefig("./figures/{}ma_sin波刺激.pdf".format(self.Is))
            plt.show()

        # return np.expand_dims(self.t,axis=1),
        v = np.expand_dims(V,axis=1)
        i_inj_values = [self.I_inj(t) for t in self.t]
        i_inj = np.expand_dims(np.array(i_inj_values),axis=1)
        ina = np.expand_dims(ina,axis=1)
        ik = np.expand_dims(ik,axis=1)
        il = np.expand_dims(il,axis=1)
        
        u = np.expand_dims(V,axis=1)
        m = np.expand_dims(m,axis=1)
        h = np.expand_dims(h,axis=1)
        n = np.expand_dims(n,axis=1)
        t = np.expand_dims(self.t,axis=1)
        
        # np.savez('g_Na=120_g_K=36_g_L=0.3.npz', t=t,v=u, m=m,h=h,n=n,i_inj=i_inj,ina=ina,ik=ik,il=il)
        filename = 'data/{}ma_sin波刺激_g_Na=120_g_K=36_g_L=0.3.npz'.format(self.Is)
        # np.savez(filename, t=t,v=u, m=m,h=h,n=n,i_inj=i_inj,ina=ina,ik=ik,il=il)

        return  np.squeeze(self.t), np.hstack((v, m, h, n)) ,i_inj

if __name__ == '__main__':
    # for i in np.arange(10,80,0.5):
    i=10
    runner = HodgkinHuxley(t_n=100,g_Na=120, g_K=36, g_L=0.3,delta_t=0.01,Is=i,Is_type="hengding")
    observe_t,ob_y ,i_inj= runner.Main(plot=True)
