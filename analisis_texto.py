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
base=pd.read_excel("D:/Mi unidad/Uba/Maestria datos financieros/Metodos datos no estructurados/Clase 3/Base canciones.xlsx")
## separo estrofas que quedaron juntas
def agregar_espacios(texto):
    return re.sub(r'([A-Z])', r' \1', texto)

base['Letra']= base['Letra'].apply(agregar_espacios)
# qutar acentos
def quitar_acentos(texto):
    return unidecode(texto)

# Aplicar la función a la columna 'texto'
base['Letra'] = base['Letra'].apply(quitar_acentos)
base= base[base['Artista']!='Los Redonditos de Ricota']
## Genero bases individuales por artista
base_mana=base[base['Artista']=='Mana']
base_mana=base_mana['Letra']
base_duki=base[base['Artista']=='Duki']
base_duki=base_duki['Letra']
#base_redonditos=base[base['Artista']=='Los Redonditos de Ricota']
#base_redonditos=base_redonditos['Letra']
### Tokenizar
nltk.download('punkt')
nltk.download('stopwords')
#redonditos_tokens = base_redonditos.apply(lambda x: word_tokenize(x.lower()))
mana_tokens=base_mana.apply(lambda x: word_tokenize(x.lower()))
duki_tokens=base_duki.apply(lambda x: word_tokenize(x.lower()))
#%%###Elimino stopwords
mi_stopwords= ['tu','no','me','o','e','por',',','!','de','.','y','la',
               'el','que','a','un','en','(',')','?','!','lo','la','los',
               'las','es','...','del','con','se','mi','te',"yo", "tu",'como',
               "el", "ella", "nosotros", "ellos", "ellas",'pero','para',
               "usted", "ustedes", "mi", "ti", "si", "?",'s','d','u','¡',
               "¿", "[", "]", "'", "r", "n", "que", "cual", "cuanto", "cuantos",
               "cuyo", "cuyos", "quienes", "cuales",'/','un','una','sus','su']
# Elimina las stopwords de cada lista de tokens
#redonditos_tokens= redonditos_tokens.apply(lambda tokens: [token for token in tokens if token.lower() not in mi_stopwords])
mana_tokens=mana_tokens.apply(lambda tokens: [token for token in tokens if token.lower() not in mi_stopwords])
duki_tokens=duki_tokens.apply(lambda tokens: [token for token in tokens if token.lower() not in mi_stopwords])
#%%Lematizo

#%%
## RECUENTOS 
#all_tokens_redonditos = redonditos_tokens.sum()
all_tokens_mana = mana_tokens.sum()
all_tokens_duki= duki_tokens.sum()
# Contar la frecuencia de cada palabra
#conteo_redonditos = pd.Series(all_tokens_redonditos).value_counts()
conteo_mana = pd.Series(all_tokens_mana).value_counts()
conteo_duki = pd.Series(all_tokens_duki).value_counts()
# Opcional: convertir los resultados en un DataFrame de pandas
#conteo_redonditos = pd.DataFrame({'Palabra': conteo_redonditos.index, 'Frecuencia': conteo_redonditos.values})
conteo_mana = pd.DataFrame({'Palabra': conteo_mana.index, 'Frecuencia': conteo_mana.values})
conteo_duki = pd.DataFrame({'Palabra': conteo_duki.index, 'Frecuencia': conteo_duki.values})
#%%Nube de palabras
#Duki
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# Generate the word cloud data
wordcloud_duki = WordCloud(width=800, height=400, background_color='black').generate(' '.join(all_tokens_duki))

# Display the word cloud using matplotlib
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud_duki, interpolation='bilinear')
plt.axis('off')  # Turn off the axis
plt.show()
#Redonditos
#wordcloud_redonditos = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_tokens_redonditos))

# Display the word cloud using matplotlib
#plt.figure(figsize=(15, 10))
#plt.imshow(wordcloud_redonditos, interpolation='bilinear')
#plt.axis('off')  # Turn off the axis
#plt.show()
#mana
wordcloud_mana = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_tokens_mana))

# Display the word cloud using matplotlib
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud_mana, interpolation='bilinear')
plt.axis('off')  # Turn off the axis
plt.show()
#%%
# Cargar la imagen
imagen = plt.imread('C:/Users/nlenardon/Downloads/redonditos.png')  # Reemplaza 'ruta/a/tu/imagen.jpg' con la ruta a tu imagen

# Crear una figura de Matplotlib
plt.figure(figsize=(15, 10))

# Colocar la imagen en el margen superior izquierdo
plt.figimage(imagen, xo=0, yo=plt.gca().get_window_extent().height - imagen.shape[0]-100)

# Crear el WordCloud
wordcloud_mana = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_tokens_mana))

# Mostrar el WordCloud
plt.imshow(wordcloud_mana, interpolation='bilinear')
plt.axis('off')  # Ocultar los ejes

# Mostrar la figura
plt.show()

#%%CLASIFICACION
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
## divido datos
X_train, X_test, y_train, y_test = train_test_split(base['Letra'], base['Artista'], test_size=0.2, random_state=36)
# Vectorizo letsa
vectorizer = TfidfVectorizer(max_features=400)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
## Entreno
bayes_clasificador = MultinomialNB()
bayes_clasificador.fit(X_train_vec, y_train)
##Predigo
y_pred = bayes_clasificador.predict(X_test_vec)
print(y_pred)
# Evaluo modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
#%%
###Predigo
X_test=["baby moveme ese culo"]
X_test_vec = vectorizer.transform(X_test)
y_pred = bayes_clasificador.predict(X_test_vec)
print(y_pred)