import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from unidecode import unidecode
#%% Preprocesamiento
## Cargo base de canciones
base=pd.read_excel("G:/Mi unidad/Uba/Maestria datos financieros/Metodos datos no estructurados/Clase 3/Base canciones.xlsx")
## separo estrofas que quedaron juntas
def agregar_espacios(texto):
    return re.sub(r'([A-Z])', r' \1', texto)

base['Letra']= base['Letra'].apply(agregar_espacios)
## Genero bases individuales por artista
base_mana=base[base['Artista']=='Mana']
base_mana=base_mana['Letra']
base_duki=base[base['Artista']=='Duki']
base_duki=base_duki['Letra']
base_redonditos=base[base['Artista']=='Los Redonditos de Ricota']
base_redonditos=base_redonditos['Letra']
### Tokenizar
nltk.download('punkt')
nltk.download('stopwords')
redonditos_tokens = base_redonditos.apply(lambda x: word_tokenize(x.lower()))
mana_tokens=base_mana.apply(lambda x: word_tokenize(x.lower()))
duki_tokens=base_duki.apply(lambda x: word_tokenize(x.lower()))
#%%###Elimino stopwords
mi_stopwords= ['tu',',','!','de','.','y','la','el','que','a','un','en','(',')','?','!','lo','la','los','las','es','...','del','con','se','mi','te',"yo", "tu", "el", "ella", "nosotros", "vosotros", "ellos", "ellas", "usted", "ustedes", "mi", "ti", "si", "consigo", "nosotros/as", "vosotros/as", "mi mismo/a", "ti mismo/a", "si mismo/a", "quien", "que", "cual", "cuanto/a", "cuantos/as", "cuyo/a", "cuyos/as", "quienes", "cuales"]
# Elimina las stopwords de cada lista de tokens
redonditos_tokens= redonditos_tokens.apply(lambda tokens: [token for token in tokens if token.lower() not in mi_stopwords])
mana_tokens=mana_tokens.apply(lambda tokens: [token for token in tokens if token.lower() not in mi_stopwords])
duki_tokens=duki_tokens.apply(lambda tokens: [token for token in tokens if token.lower() not in mi_stopwords])
#%%
## RECUENTOS 
all_tokens_redonditos = redonditos_tokens.sum()
all_tokens_mana = mana_tokens.sum()
all_tokens_duki= duki_tokens.sum()
# Contar la frecuencia de cada palabra
conteo_redonditos = pd.Series(all_tokens_redonditos).value_counts()
conteo_mana = pd.Series(all_tokens_mana).value_counts()
conteo_duki = pd.Series(all_tokens_duki).value_counts()
# Opcional: convertir los resultados en un DataFrame de pandas
conteo_redonditos = pd.DataFrame({'Palabra': conteo_redonditos.index, 'Frecuencia': conteo_redonditos.values})
conteo_mana = pd.DataFrame({'Palabra': conteo_mana.index, 'Frecuencia': conteo_mana.values})
conteo_duki = pd.DataFrame({'Palabra': conteo_duki.index, 'Frecuencia': conteo_duki.values})