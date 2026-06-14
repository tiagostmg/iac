import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import statistics

# Suprimindo possíveis warnings matemáticos
warnings.filterwarnings("ignore")

class OtimizacaoContinua:
    def __init__(self):
        # Cria a pasta para salvar os resultados, se não existir
        self.pasta_resultados = 'plot_resultados'
        os.makedirs(self.pasta_resultados, exist_ok=True)
        
        # Configurações para cada uma das 6 funções
        self.configs = {
            1: {'tipo': 'min', 'limites': [(-100.0, 100.0), (-100.0, 100.0)]},
            2: {'tipo': 'max', 'limites': [(-2.0, 4.0), (-2.0, 5.0)]},
            3: {'tipo': 'min', 'limites': [(-8.0, 8.0), (-8.0, 8.0)]},
            4: {'tipo': 'min', 'limites': [(-5.12, 5.12), (-5.12, 5.12)]},
            5: {'tipo': 'max', 'limites': [(-10.0, 10.0), (-10.0, 10.0)]},
            6: {'tipo': 'max', 'limites': [(-1.0, 3.0), (-1.0, 3.0)]}
        }
        
    def calcular_funcao(self, x, num_funcao):
        """Aplica as equações matemáticas (funciona tanto com escalares quanto matrizes/meshgrids)."""
        x1, x2 = x[0], x[1]
        
        if num_funcao == 1:
            return x1**2 + x2**2
        elif num_funcao == 2:
            return np.exp(-(x1**2 + x2**2)) + 2 * np.exp(-((x1 - 1.7)**2 + (x2 - 1.7)**2))
        elif num_funcao == 3:
            term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))
            term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
            return term1 + term2 + 20 + np.exp(1)
        elif num_funcao == 4:
            return (x1**2 - 10 * np.cos(2 * np.pi * x1) + 10) + (x2**2 - 10 * np.cos(2 * np.pi * x2) + 10)
        elif num_funcao == 5:
            return (x1 * np.cos(x1)) / 20 + 2 * np.exp(-(x1**2) - (x2 - 1)**2) + 0.01 * x1 * x2
        elif num_funcao == 6:
            return x1 * np.sin(4 * np.pi * x1) - x2 * np.sin(4 * np.pi * x2 + np.pi) + 1
        else:
            raise ValueError("Função inválida.")

    def verificar_melhoria(self, f_cand, f_opt, tipo):
        if tipo == 'min':
            return f_cand < f_opt
        return f_cand > f_opt

    def aplicar_limites(self, x, limites):
        x[0] = np.clip(x[0], limites[0][0], limites[0][1])
        x[1] = np.clip(x[1], limites[1][0], limites[1][1])
        return x

    def hill_climbing(self, num_funcao, epsilon=0.1, max_iteracoes=1000, parada_t=100):
        config = self.configs[num_funcao]
        limites = config['limites']
        tipo = config['tipo']
        
        x_opt = np.array([limites[0][0], limites[1][0]])
        f_opt = self.calcular_funcao(x_opt, num_funcao)
        
        # Variáveis de rastreamento para os gráficos
        historico_f = [f_opt]
        historico_x = [x_opt.copy()]
        
        iter_sem_melhoria = 0
        for _ in range(max_iteracoes):
            x_cand = np.random.uniform(x_opt - epsilon, x_opt + epsilon)
            x_cand = self.aplicar_limites(x_cand, limites)
            f_cand = self.calcular_funcao(x_cand, num_funcao)
            
            if self.verificar_melhoria(f_cand, f_opt, tipo):
                x_opt = x_cand
                f_opt = f_cand
                iter_sem_melhoria = 0
            else:
                iter_sem_melhoria += 1
            
            historico_f.append(f_opt)
            historico_x.append(x_opt.copy())
            
            if iter_sem_melhoria >= parada_t:
                break
                
        return f_opt, x_opt, historico_f, historico_x

    def local_random_search(self, num_funcao, sigma=0.2, max_iteracoes=1000, parada_t=100):
        config = self.configs[num_funcao]
        limites = config['limites']
        tipo = config['tipo']
        
        x_opt = np.array([np.random.uniform(limites[0][0], limites[0][1]),
                          np.random.uniform(limites[1][0], limites[1][1])])
        f_opt = self.calcular_funcao(x_opt, num_funcao)
        
        historico_f = [f_opt]
        historico_x = [x_opt.copy()]
        
        iter_sem_melhoria = 0
        for _ in range(max_iteracoes):
            ruido = np.random.normal(0, sigma, size=2)
            x_cand = x_opt + ruido
            x_cand = self.aplicar_limites(x_cand, limites)
            f_cand = self.calcular_funcao(x_cand, num_funcao)
            
            if self.verificar_melhoria(f_cand, f_opt, tipo):
                x_opt = x_cand
                f_opt = f_cand
                iter_sem_melhoria = 0
            else:
                iter_sem_melhoria += 1
                
            historico_f.append(f_opt)
            historico_x.append(x_opt.copy())
            
            if iter_sem_melhoria >= parada_t:
                break
                
        return f_opt, x_opt, historico_f, historico_x

    def global_random_search(self, num_funcao, max_iteracoes=1000, parada_t=100):
        config = self.configs[num_funcao]
        limites = config['limites']
        tipo = config['tipo']
        
        x_opt = np.array([np.random.uniform(limites[0][0], limites[0][1]),
                          np.random.uniform(limites[1][0], limites[1][1])])
        f_opt = self.calcular_funcao(x_opt, num_funcao)
        
        historico_f = [f_opt]
        historico_x = [x_opt.copy()]
        
        iter_sem_melhoria = 0
        for _ in range(max_iteracoes):
            x_cand = np.array([np.random.uniform(limites[0][0], limites[0][1]),
                               np.random.uniform(limites[1][0], limites[1][1])])
            f_cand = self.calcular_funcao(x_cand, num_funcao)
            
            if self.verificar_melhoria(f_cand, f_opt, tipo):
                x_opt = x_cand
                f_opt = f_cand
                iter_sem_melhoria = 0
            else:
                iter_sem_melhoria += 1
                
            historico_f.append(f_opt)
            historico_x.append(x_opt.copy())
            
            if iter_sem_melhoria >= parada_t:
                break
                
        return f_opt, x_opt, historico_f, historico_x

    def gerar_graficos_rodada(self, historico_f, historico_x, algoritmo, num_funcao):
        """Gera e salva os gráficos de aptidão e de superfície 3D na pasta especificada."""
        
        # 1. Gráfico de Histórico de Aptidão (Convergência)
        plt.figure(figsize=(8, 5))
        plt.plot(historico_f, color='blue', linewidth=2)
        plt.title(f'Histórico de Aptidão - {algoritmo.upper()} (Função {num_funcao})')
        plt.xlabel('Iterações')
        plt.ylabel('Valor da Função (Aptidão)')
        plt.grid(True, linestyle='--', alpha=0.7)
        caminho_aptidao = os.path.join(self.pasta_resultados, f'aptidao_{algoritmo}_f{num_funcao}.png')
        plt.savefig(caminho_aptidao, bbox_inches='tight')
        plt.close()
        
        # 2. Gráfico 3D da Superfície e do Caminho Percorrido
        limites = self.configs[num_funcao]['limites']
        x1_vals = np.linspace(limites[0][0], limites[0][1], 100)
        x2_vals = np.linspace(limites[1][0], limites[1][1], 100)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        Z = self.calcular_funcao([X1, X2], num_funcao)
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plotando a superfície (suave e semi-transparente)
        ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.5, edgecolor='none')
        
        # Extraindo histórico de pontos X e Z
        hx = np.array(historico_x)
        hz = [self.calcular_funcao(x, num_funcao) for x in hx]
        
        # Plotando o caminho percorrido pelo algoritmo
        ax.plot(hx[:,0], hx[:,1], hz, color='red', marker='.', markersize=4, linewidth=1.5, label='Caminho')
        
        # Destacando Início e Fim
        ax.scatter(hx[0,0], hx[0,1], hz[0], color='blue', s=100, label='Início', zorder=5)
        ax.scatter(hx[-1,0], hx[-1,1], hz[-1], color='orange', s=200, marker='*', label='Ótimo Encontrado', zorder=5)
        
        ax.set_title(f'Superfície e Busca - {algoritmo.upper()} (Função {num_funcao})')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('f(x)')
        ax.legend()
        
        caminho_3d = os.path.join(self.pasta_resultados, f'superficie_{algoritmo}_f{num_funcao}.png')
        plt.savefig(caminho_3d, bbox_inches='tight')
        plt.close()

    def executar_experimento(self, num_funcao, algoritmo='hill_climbing', rodadas=100, **kwargs):
        resultados_f = []
        
        for r in range(rodadas):
            if algoritmo == 'hill_climbing':
                f_opt, _, hist_f, hist_x = self.hill_climbing(num_funcao, **kwargs)
            elif algoritmo == 'lrs':
                f_opt, _, hist_f, hist_x = self.local_random_search(num_funcao, **kwargs)
            elif algoritmo == 'grs':
                f_opt, _, hist_f, hist_x = self.global_random_search(num_funcao, **kwargs)
            
            resultados_f.append(round(f_opt, 4))
            
            # Condição para salvar os gráficos apenas na PRIMEIRA RODADA (índice 0)
            if r == 0:
                print(f"[{algoritmo.upper()}] Salvando gráficos da 1ª rodada na pasta '{self.pasta_resultados}'...")
                self.gerar_graficos_rodada(hist_f, hist_x, algoritmo, num_funcao)
            
        resultados_moda = [round(val, 1) for val in resultados_f]
        moda = statistics.mode(resultados_moda)
        media = round(statistics.mean(resultados_f), 4)

        print(f"[{algoritmo.upper()} - Função {num_funcao}] Resultados após {rodadas} rodadas:")
        print(f"  -> Moda:  {moda}")
        print(f"  -> Média: {media}\n")
        
        return moda, media

# =========================================================
# Exemplo de Execução
# =========================================================
if __name__ == "__main__":
    otimizador = OtimizacaoContinua()
    
    funcao_escolhida = 6 
    rodadas = 100
    
    print(f"--- Iniciando testes para a Função {funcao_escolhida} ---\n")
    
    # Executa os métodos e gera as imagens da primeira rodada de cada um deles
    otimizador.executar_experimento(num_funcao=funcao_escolhida, algoritmo='hill_climbing', rodadas=rodadas, epsilon=2)
    otimizador.executar_experimento(num_funcao=funcao_escolhida, algoritmo='lrs', rodadas=rodadas, sigma=0.6)
    otimizador.executar_experimento(num_funcao=funcao_escolhida, algoritmo='grs', rodadas=rodadas)