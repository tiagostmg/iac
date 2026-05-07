import matplotlib.pyplot as plt
import numpy as np


class Adaline:
    def __init__(self,X_train, y_train, learning_rate, precision, max_epochs):
        self.d = y_train
        self.p, self.N = X_train.shape
        self.w = np.random.random_sample((self.p+1,1))-.5
        self.line = None
        self.X_train = np.vstack((
            -np.ones((1,self.N)), X_train
        ))
        self.lr = learning_rate
        self.pr = precision
        self.max_epochs = max_epochs


        fig = plt.figure()
        self.ax = fig.add_subplot(1,1,1)    
        self.ax.scatter(X_train[0,y_train[0,:]==1],
                           X_train[1,y_train[0,:]==1],c='pink',
                           edgecolors='k')
        self.ax.scatter(X_train[0,y_train[0,:]==-1],
                           X_train[1,y_train[0,:]==-1],c='blue',
                           edgecolors='k')
        self.ax.set_xlim(-.5,6.5)
        self.ax.set_ylim(-.5,6.5)
        self.x1 = np.linspace(-1,7)
        self.plot_2dline()
    

    def eqm(self):
        eqm = 0
        for k in range(self.N):
            x_k = self.X_train[:,k].reshape(self.p+1,1)
            u_k = (self.w.T @ x_k)[0,0]
            d_k = self.d[0,k]
            eqm += (d_k - u_k)**2
        return eqm/(2*self.N)
        
    def fit(self):
        epochs = 0
        EQM1 = 1
        EQM2 = 0
        historico_eqm = []
        while epochs < self.max_epochs and abs(EQM1 - EQM2)>self.pr:
            EQM1 = self.eqm()
            historico_eqm.append(EQM1)
            for k in range(self.N):
                x_k = self.X_train[:,k].reshape(self.p+1,1)
                u_k = (self.w.T @ x_k)[0,0]
                d_k = self.d[0,k]
                e_k = d_k - u_k
                self.w = self.w + self.lr * e_k * x_k
            # plt.pause(.01)
            self.plot_2dline()
            EQM2 = self.eqm()
            epochs+=1
            self.ax.set_title(f"Época: {epochs}")
        self.plot_2dline(c='purple',lw=4)
        plt.figure(2)
        plt.plot(historico_eqm)
        plt.xlabel("Épocas")
        plt.ylabel("EQM")
        plt.title("Curva de aprendizado do modelo")
        plt.show()
    def predict(self):
        ...

    def plot_2dline(self,c = 'g', lw = 1):
        x2 = -self.w[1,0]/self.w[2,0]*self.x1 + self.w[0,0]/self.w[2,0]
        x2 = np.nan_to_num(x2)
        if self.line != None:
            self.line[0].remove()
        self.line = self.ax.plot(self.x1,x2,c=c, lw = lw)
