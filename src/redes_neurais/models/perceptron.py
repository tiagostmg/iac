import matplotlib.pyplot as plt
import numpy as np


class Perceptron:
    def __init__(self,X_train, y_train, learning_rate):

        self.d = y_train
        self.lr = learning_rate
        self.p,self.N = X_train.shape
        self.X_train = np.vstack((
            -np.ones((1,self.N)),
            X_train
        ))
        # self.w = np.zeros((self.p+1,1))
        self.w = np.random.random_sample((self.p+1,1))-.5
        self.line2d = None
        self.line3d = None
        fig = plt.figure()
        self.ax_2d = fig.add_subplot(1,2,1)    
        self.ax_2d.scatter(X_train[0,y_train[0,:]==1],
                           X_train[1,y_train[0,:]==1],c='pink',
                           edgecolors='k')
        self.ax_2d.scatter(X_train[0,y_train[0,:]==-1],
                           X_train[1,y_train[0,:]==-1],c='blue',
                           edgecolors='k')
        self.ax_2d.set_xlim(-.5,6.5)
        self.ax_2d.set_ylim(-.5,6.5)
        self.x1 = np.linspace(-1,7)
        self.plot_2dline()
        self.ax_3d = fig.add_subplot(1,2,2, projection='3d')
        self.ax_3d.plot([-5,5],[0,0],[0,0], c = 'k',alpha=.8)
        self.ax_3d.plot([0,0],[-5,5],[0,0], c = 'k',alpha=.8)
        self.ax_3d.plot([0,0],[0,0],[-5,5], c = 'k',alpha=.8)

        self.ax_3d.set_xlim(-2,2)
        self.ax_3d.set_ylim(-2,2)
        self.ax_3d.set_zlim(-2,2)

     


    def plot_2dline(self,c = 'g'):
        x2 = -self.w[1,0]/self.w[2,0]*self.x1 + self.w[0,0]/self.w[2,0]
        x2 = np.nan_to_num(x2)
        if self.line2d != None:
            self.line2d[0].remove()
        self.line2d = self.ax_2d.plot(self.x1,x2,c=c)

    def plot3d_line(self,x):
        if self.line3d != None:
            self.line3d[0].remove()
            self.line3d2[0].remove()
            self.dot1.remove()
            self.dot2.remove()

        self.line3d = self.ax_3d.plot([x[0,0],0],[x[1,0],0],[x[2,0],0],lw=3,c='r')
        self.dot1 = self.ax_3d.scatter(x[0,0],x[1,0],x[2,0],s=70,color='r')

        self.line3d2 = self.ax_3d.plot([self.w[0,0],0],[self.w[1,0],0],[self.w[2,0],0],lw=3,c='#BD8A2D')
        self.dot2 = self.ax_3d.scatter(self.w[0,0],self.w[1,0],self.w[2,0],s=70,color='#BD8A2D')


    def activation_function(self, u):
        return 1 if u>=0 else -1
    def fit(self):
        error = True
        epochs = 0
        while error:
            error = False
            for k in range(self.N):
                x_k = self.X_train[:,k].reshape(self.p+1,1)
                d_k = self.d[0,k]
                u_k = (self.w.T@x_k)[0,0]
                y_k = self.activation_function(u_k)
                e_k = d_k - y_k
                if e_k != 0:
                    error = True
              
                if u_k >= 0:
                    self.ax_3d.set_title(f"u({k}) = {u_k:.5f}, "+r"$\alpha \leq 90^\circ$")
                else:
                    self.ax_3d.set_title(f"u({k}) = {u_k:.5f}, "+r"$\alpha > 90^\circ$")
                self.plot_2dline(c='k')
                self.ax_2d.set_title(f"u({k})= {y_k}, d({k})={d_k}, e({k})= {e_k} ")
                self.plot3d_line(x_k)
                
                self.w = self.w + self.lr * e_k * x_k
                u_k = (self.w.T@x_k)[0,0]
                if u_k >= 0:
                    self.ax_3d.set_title(f"u({k}) = {u_k:.5f}, "+r"$\alpha \leq 90^\circ$")
                else:
                    self.ax_3d.set_title(f"u({k}) = {u_k:.5f}, "+r"$\alpha > 90^\circ$")
                self.plot_2dline(c='k')
                self.ax_2d.set_title(f"u({k})= {y_k}, d({k})={d_k}, e({k})= {e_k} ")
                self.plot3d_line(x_k)
            plt.pause(.2)
            epochs+=1
        self.plot_2dline(c='purple')
        plt.show()
