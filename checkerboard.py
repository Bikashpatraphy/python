
from pythtb import *
import numpy as np
import matplotlib.pyplot as plt
import array 


#define lattice vectors
def set_model(delta, t_0, tprime):
    lat = [[1.0, 0.0],[0.0, 1.0]]
    # define orbital position
    orb =[[0.0, 0.0], [0.5, 0.5]]
    my_model =  tb_model(2, 2, lat, orb)
    # set onsite energies
    my_model.set_onsite([-delta, delta])
    # set hopping
    my_model.set_hop(-tprime, 1, 0, [0, 0])
    my_model.set_hop(tprime, 1, 0, [1, 1])
    my_model.set_hop(-tprime*1j, 1, 0, [1, 0])
    my_model.set_hop(tprime*1j, 1, 0, [0, 1])
    my_model.set_hop(-t_0, 0, 0, [1, 0])
    my_model.set_hop(-t_0, 0, 0, [0, 1])
    my_model.set_hop(t_0, 1, 1, [1, 0])
    my_model.set_hop(t_0, 1, 1, [0, 1])
    return my_model
#my_model.display()
#fig, ax =my_model.visualize(0, 1)


# In[243]:


#delta =1.0
t_0 = 1.0
tprime = 0.8
labs = [['(a)','(b)','(c)'],['(d)','(e)','(f)']]

fig, ax = plt.subplots(2,3, figsize=(15,8))
for j2, delta in enumerate([5.0, 1.0]):
    my_model = set_model(delta, t_0, tprime)
    path = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.5], [0.0, 0.0]]
    label = (r'$\Gamma $', r'$ X $', r'$ M $', r'$ \Gamma $')
    k_vec, k_dist, k_node = my_model.k_path(path, 301)
    eval = my_model.solve_all(k_vec)
    ax[j2,0].set_xlim(k_node[0],k_node[-1])
    ax[j2,0].set_xticks(k_node)
    ax[j2,0].set_xticklabels(label)
    for n in range(len(k_node)):
        ax[j2, 0].axvline(x=k_node[n], linewidth=0.5, color='k')
    for i in range(2):
        ax[j2,0].plot(k_dist,eval[i])

    ax[j2,0].set_title(labs[j2][0])  
    ax[j2,0].set_xlabel('Path in k space')
    ax[j2,0].set_ylabel('Band Energy')
    nk =61
    dk = 2*np.pi/(nk - 1)
    my_array = wf_array(my_model,[nk, nk])
    my_array.solve_on_grid([0. , 0.])
    bcurv = my_array.berry_flux([0], individual_phases=True)/(dk*dk)
    chern = my_array.berry_flux([0])/(2.*np.pi)
    print('chern number =', chern)
    rbar_1 = my_array.berry_phase([0], 1, contin=True)/(2.*np.pi)
    k0 = np.linspace(0., 1., nk)
    ax[j2,1].set_xlim(0., 1.0)
    ax[j2,1].set_title(labs[j2][1])      
    ax[j2,1].set_xlabel(r'$\kappa_1/2\pi$')
    ax[j2,1].set_ylabel(r'HWF centers')
    for shift in (-2., -1., 0., 1.):
        ax[j2,1].plot(k0, rbar_1+shift)
    width = 20
    nkr = 81
    ribbon_model = my_model.cut_piece(width, fin_dir=1, glue_edgs=False)
    k_vecr, k_distr, k_noder = ribbon_model.k_path('full', nkr, report=False)
    rib_eval, rib_evec = ribbon_model.solve_all(k_vecr, eig_vectors=True)
    nbands = rib_eval.shape[0]
    k0 = np.linspace(0., 1., nkr)

    ax[j2,2].set_xlim(0., 1.)
    ax[j2,2].set_title(labs[j2][2])      
    ax[j2,2].set_xlabel(r'$\kappa_1/2\pi$')
    ax[j2,2].set_ylabel(r'Edge band structure')
    for (i,kv) in enumerate(k0):
        ax[j2,2].scatter([k_vecr[i]]*nbands, rib_eval[:,i],s=0.8,color='k', marker='o')   
fig.tight_layout()    
fig.savefig('checkerboard.pdf')        





