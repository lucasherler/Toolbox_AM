# import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from _results import Result
# import ApresentacaoResultados.ImagemTabela as ImagemTabela

#matplotlib.rcParams['interactive'] == True
matplotlib.use('TkAgg')

class IMAGEMTABELA(Result):
    @staticmethod
    def getName():
        return "Imagem das Tabelas"
    
    @staticmethod
    def generate(dict_of_dataframes):
        dfcopy = dict_of_dataframes.copy()
        dfcopy.pop('parametros')
        for key in dfcopy.keys():
            df = dfcopy[key]
            df.update(df.applymap('{:,.3f}'.format))
            fig, ax = plt.subplots()
            fig.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')
            # ax.set_ylabel('aaa')
            ax.set_title('Métrica ' + key)
            ax.table(cellText=df.values, colLabels=df.columns, rowLabels = df.index, loc='center')

            fig.tight_layout()
            plt.show()
            
#df = test[metric]
# df.update(df.applymap('{:,.3f}'.format))

# # print(test[metric])
# # # from pandas.plotting import table 
# fig, ax = plt.subplots()

# # hide axes
# fig.patch.set_visible(False)
# ax.axis('off')
# ax.axis('tight')
# ax.set_ylabel('aaa')

# ax.table(cellText=test['ACCURACY'].values, colLabels=test['ACCURACY'].columns, rowLabels = test['ACCURACY'].index, loc='center')

# fig.tight_layout()
        
        