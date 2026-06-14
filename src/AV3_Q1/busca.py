import numpy as np
import matplotlib.pyplot as plt

def f(x1,x2):
    return np.exp(-(x1**2 + x2**2)) + 2*np.exp(-((x1-1.6)**2 + (x2-1.6)**2))

#restrição de domínio
inferior, superior = -2,4

fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')
x_axis = np.linspace(inferior,superior,1000)
X1, X2 = np.meshgrid(x_axis,x_axis)
ax.plot_surface(X1,X2, f(X1,X2), cmap='gray', edgecolor='g', alpha=.2,
                rstride = 40, cstride= 40,)



# Hill Climbing (Subida de Encosta)
# x_opt = np.random.uniform(inferior, superior, size=(2,))
# f_opt = f(*x_opt)

# ax.scatter(*x_opt, f_opt, s=50, c = 'r')
# plt.pause(.1)
# epsilon = 2
# max_iteracoes = 100000
# max_vizinhos = 100

# it = 0
# melhoria = True

# while it < max_iteracoes and melhoria:
#     melhoria = False
#     for j in range(max_vizinhos):
#         # perturbação do ótimo
#         x_cand = np.random.uniform(x_opt-epsilon, x_opt+epsilon)
#         mask = x_cand > superior    
#         x_cand[mask] = superior
#         mask = x_cand < inferior    
#         x_cand[mask] = inferior

#         f_cand = f(*x_cand)
#         ax.scatter(*x_opt, f_opt, s=50, c = 'b')

#         if f_cand > f_opt :
#             x_opt = x_cand
#             f_opt = f_cand
#             melhoria = True
#             ax.scatter(*x_opt, f_opt, s=50, c = 'r')
#             plt.pause(.1)
#             break
#     it+=1
# ax.scatter(*x_opt, f_opt, s=500,marker = '*', c = 'orange')

#Busca Aleatória Local
# x_opt = np.random.uniform(inferior, superior, size=(2,))
# f_opt = f(*x_opt)

# ax.scatter(*x_opt, f_opt, s=50, c = 'r')
# plt.pause(.1)
# sigma = .2
# max_iteracoes = 10000


# it = 0


# while it < max_iteracoes:
   
#     n = np.random.normal(0, sigma, size=(2,))
#     x_cand = x_opt + n
#     mask = x_cand > superior    
#     x_cand[mask] = superior
#     mask = x_cand < inferior    
#     x_cand[mask] = inferior

#     f_cand = f(*x_cand)
#     ax.scatter(*x_cand, f_cand, s=50, c = 'b')
#     plt.pause(.1)
#     if f_cand > f_opt :
#         x_opt = x_cand
#         f_opt = f_cand      
#         ax.scatter(*x_opt, f_opt, s=50, c = 'r')
#         plt.pause(.1)
        
#     it+=1
# ax.scatter(*x_opt, f_opt, s=500,marker = '*', c = 'orange')


#Busca Aleatória Global
x_opt = np.random.uniform(inferior, superior, size=(2,))
f_opt = f(*x_opt)

ax.scatter(*x_opt, f_opt, s=50, c = 'r')
plt.pause(.1)

max_iteracoes = 10000


it = 0


while it < max_iteracoes:
   
   
    x_cand = np.random.uniform(inferior, superior, size=(2,))
  
    f_cand = f(*x_cand)
    ax.scatter(*x_cand, f_cand, s=50, c = 'b')
    plt.pause(.1)
    if f_cand > f_opt :
        x_opt = x_cand
        f_opt = f_cand      
        ax.scatter(*x_opt, f_opt, s=50, c = 'r')
        plt.pause(.1)
        
    it+=1
ax.scatter(*x_opt, f_opt, s=500,marker = '*', c = 'orange')




plt.show()
