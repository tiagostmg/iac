import matplotlib.pyplot as plt
import numpy as np


class MultilayerPerceptron:
    def __init__(self, topology, X_train, Y_train, learning_rate, max_epochs, precision):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.pr = precision
        self.topology = topology
        self.D = Y_train
        m = Y_train.shape[0]
        self.p, self.N = X_train.shape
        self.topology.append(m)
        self.X_train = np.vstack((
            -np.ones((1,self.N)), X_train
        ))
        self.D = Y_train
        self.W = []
        for i,q in enumerate(self.topology):
            if i == 0:
                W = np.random.random_sample((q, self.p+1))-.5
            else:
                W = np.random.random_sample((q, self.topology[i-1]+1))-.5
            self.W.append(W)
            
        self.u = [None]*len(self.W)
        self.y = [None]*len(self.W)
        self.delta = [None]*len(self.W)

        #  NÃO FAZ PARTE DO MODELO

        fig = plt.figure(1)
        self.ax = fig.add_subplot(1,1,1)
        self.ax.scatter(X_train[0, Y_train[0,:]==1], X_train[1, Y_train[0,:]==1],c = 'pink',edgecolors='k')
        self.ax.scatter(X_train[0, Y_train[0,:]==-1], X_train[1, Y_train[0,:]==-1],c = 'purple',edgecolors='k')
        self.ax.set_xlim(-1.1,1.1)
        self.ax.set_ylim(-1.1,1.1)
        x1 = np.linspace(-1.2,1.2,200)
        self.X1, self.X2 = np.meshgrid(x1,x1)
        self.X_plot = np.vstack((
            -np.ones((1, len(x1)*len(x1))),
            np.ravel(self.X1),
            np.ravel(self.X2)
        ))
        print(self.X_plot.shape)
        self.contour = None
        # self.ax.scatter(self.X_plot[1,:],self.X_plot[2,:],c='g')
        # plt.show()

    def plot_contour(self,e):

        prediction = np.empty((1,0))
        N = self.X_plot.shape[1]
        for k in range(N):
            x_k = self.X_plot[:,k].reshape(self.p+1,1)
            self.forward(x_k)

            prediction = np.hstack((
                prediction, np.ones((1,1)) if self.y[-1] >= 0 else -np.ones((1,1))
            ))
        if self.contour != None:
            self.contour.remove()
        self.contour = self.ax.contourf(self.X1, self.X2, prediction.reshape(self.X1.shape),alpha=.3)
        self.ax.set_title(f"Época: {e}")
        plt.pause(.001)
    def g(self,u):
        s = np.exp(-u)
        return (1-s)/(1+s)
    
    def g_d(self,u):
        s = self.g(u)
        return .5*(1-s**2)

    def eqm(self):
        eqm = 0
        for k in range(self.N):
            x_k = self.X_train[:,k].reshape(self.p+1,1)
            self.forward(x_k)
            d_k = self.D[:,k].reshape(self.topology[-1],1)
            eqm += np.sum(d_k - self.y[-1])**2
        return eqm/(2*self.N)
    def forward(self,x):
        for j,W in enumerate(self.W):
            if j == 0:
                self.u[j] = W@x
            else:
                yb = np.vstack((
                    -np.ones((1,1)), self.y[j-1]
                )) 
                self.u[j] = W @ yb
                
            self.y[j] = self.g(self.u[j])

    def backward(self, x, d):

        for j in range(len(self.W)-1, -1, -1):
            if j == len(self.W)-1:
                self.delta[j] = self.g_d(self.u[j]) * (d - self.y[-1])
                yb = np.vstack((
                    -np.ones((1,1)), self.y[j-1]
                ))
                self.W[j] = self.W[j] + self.lr*self.delta[j] @ yb.T
            elif j == 0:
                Wnb = ((self.W[j+1])[:,1:]).T
                self.delta[j] = self.g_d(self.u[j]) * (Wnb@self.delta[j+1])
                self.W[j] = self.W[j] + self.lr*self.delta[j] @ x.T
            else:
                Wnb = ((self.W[j+1])[:,1:]).T
                self.delta[j] = self.g_d(self.u[j]) * (Wnb@self.delta[j+1])
                yb = np.vstack((
                    -np.ones((1,1)), self.y[j-1]
                ))
                self.W[j] = self.W[j] + self.lr*self.delta[j] @ yb.T



       
    def fit(self):
        epochs = 0
        self.plot_contour(epochs)
        EQM = self.eqm()
        while epochs < self.max_epochs and EQM > self.pr:
            for k in range(self.N):
                x_k = self.X_train[:,k].reshape(self.p+1,1)
                d_k = self.D[:,k].reshape(self.topology[-1],1)
                self.forward(x_k)
                self.backward(x_k, d_k)
                
            epochs+=1
            if epochs % 25 == 0:
                self.plot_contour(epochs)
            # print(EQM)
            EQM = self.eqm()

        


    def predict(self):
        pass