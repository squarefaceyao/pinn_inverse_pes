# pinn_inverse_pes


## We will solve a simple ODE system:


$$ {\frac{dV}{dt}}=10- {G_{Na}m^3h(V-50)} - {G_{K}n^4(V+77)} - {G_{L}(V+54.387)}$$ 

$$ {\frac{dm}{dt}}=\left(\frac{0.1{(V+40)}}{1-e^\frac{-V-40}{10}}\right)(1-m) - \left(4e^{\frac{-V-65}{18}}\right)m $$ 

$$ \frac{dh}{dt}= {\left(0.07e^{\frac{-V-65}{20}}\right)(1-h)} - \left(\frac{1}{1+e^\frac{-V-35}{10}}\right)h $$ 

$$  \frac{dn}{dt}= {\left(\frac{0.01(V+55)}{1-e^\frac{-V-55}{10}}\right)}(1-n) - \left(0.125e^{\frac{-V-65}{80}}\right)n$$ 

$$\qquad \text{where} \quad t \in [0,7],$$

with the initial conditions  

$$ V(0) = -65,  m(0) = 0.05 ,  h(0) = 0.6 ,  n(0) = 0.32 $$

The reference solution is [here](https://drive.google.com/file/d/1gtq0Gwi06170MOAGoxOLPGT5pJ54Jb2d/view?usp=sharing), where the parameters $G_{na},G_{k},G_{L}$ are gated variables and whose true values are 120, 36, and 0.3, respectivly. 


