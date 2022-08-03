# import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from _results import Result
import pandas as pd
# import ApresentacaoResultados.ImagemTabela as ImagemTabela

matplotlib.use('TkAgg')

class MOSTRARPARAMETROS(Result):
    @staticmethod
    def getName():
        return "Mostrar Parâmetros"
    
    @staticmethod
    def generate(dict_of_dataframes):
        params = dict_of_dataframes['parametros']
        #print(params)
        if len(params) != 0 and len(params[list(params.keys())[0]]) != 0: #Segundo termo só é executado se o primeiro não passar
            df = pd.DataFrame.from_dict(params,orient='index')
            fig, ax = plt.subplots()
            fig.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')
            # ax.set_ylabel('aaa')
            ax.set_title('Parâmetros')
            print(df)
            ax.table(cellText=df.values, colLabels=df.columns, rowLabels = df.index, loc='center')
            plt.show()
        # print(params)
        
            
            
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
        
if __name__ == '__main__':
    # df = {}
    # df['parametros'] = {'KNN': {'K': 5},'RELM': {'seed':123,'reg':456}}
    # print(df)
    # MOSTRARPARAMETROS.generate(df)
    
    pass
