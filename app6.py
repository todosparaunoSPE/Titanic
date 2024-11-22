# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:43:09 2024

@author: jperezr
"""


import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Título de la aplicación
st.title('Predicción de Supervivencia del Titanic')

# Subir archivos CSV
train_file = st.file_uploader("Sube el archivo de entrenamiento (train.csv)", type=["csv"])
test_file = st.file_uploader("Sube el archivo de prueba (test.csv)", type=["csv"])

# Verificar si se han subido los archivos
if train_file is not None and test_file is not None:
    # Cargar los datos
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    # Mostrar los primeros registros de los datos
    st.subheader("Datos de Entrenamiento:")
    st.write(train_data)  # Muestra todo el DataFrame de entrenamiento
    
    st.subheader("Datos de Prueba:")
    st.write(test_data)  # Muestra todo el DataFrame de prueba
    
    # Visualización de la distribución de los sobrevivientes por género
    st.subheader("Distribución de Supervivientes por Género")
    fig, ax = plt.subplots()
    sns.countplot(data=train_data, x='Survived', hue='Sex', ax=ax)
    st.pyplot(fig)
    
    # Mostrar el porcentaje de hombres y mujeres
    male, female = train_data.Sex.value_counts()
    st.write(f"Porcentaje de Hombres: {round(male/(male+female)*100)}%")
    st.write(f"Porcentaje de Mujeres: {round(female/(male+female)*100)}%")
    
    # Visualización de la distribución de supervivientes por clase
    st.subheader("Distribución de Supervivientes por Clase")
    fig, ax = plt.subplots()
    sns.countplot(data=train_data, x='Survived', hue='Pclass', ax=ax)
    st.pyplot(fig)
    
    # Mostrar porcentaje de supervivientes por clase
    class_counts = train_data.Pclass.value_counts()
    total_class = class_counts.sum()
    for _class, count in class_counts.items():
        percentage = (count / total_class) * 100
        st.write(f"Porcentaje con clase {_class}: {round(percentage, 2)}%")
    
    # Visualización de la distribución de supervivientes por cantidad de hijos
    st.subheader("Distribución de Supervivientes por Cantidad de Hijos")
    fig, ax = plt.subplots()
    sns.countplot(data=train_data, x='Survived', hue='Parch', ax=ax)
    st.pyplot(fig)
    
    # Mostrar porcentaje de supervivientes con diferentes cantidades de hijos
    parch_counts = train_data.Parch.value_counts()
    total_parch = parch_counts.sum()
    for parch, count in parch_counts.items():
        percentage = (count / total_parch) * 100
        st.write(f"Porcentaje con {parch} hijos a bordo: {round(percentage, 2)}%")
    
    # Preprocesamiento de los datos
    train_data.replace('male', 1, inplace=True)
    train_data.replace('female', 0, inplace=True)
    
    # Reemplazar valores nulos en la columna 'Age'
    not_zero = ['Age']
    for column in not_zero:
        mean = int(train_data[column].mean(skipna=True))
        train_data[column] = train_data[column].replace(np.nan, mean)
    
    # Selección de características y variable objetivo
    X = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    y = train_data.Survived
    
    # División de los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Entrenamiento del modelo de árbol de decisión
    model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
    model.fit(X_train, y_train)
    
    # Mostrar el gráfico del árbol de decisión con mayor ancho
    st.subheader("Gráfico del Árbol de Decisión")
    fig, ax = plt.subplots(figsize=(25, 10))  # Aumento el ancho a 15 y la altura a 10
    plot_tree(model, filled=True, ax=ax, feature_names=X.columns, class_names=["No sobrevivió", "Sobrevivió"], fontsize=12)
    st.pyplot(fig)
    
    # Predicción de la supervivencia
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    precision = precision_score(y_pred, y_test)
    recall = recall_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    st.subheader(f"Precisión del modelo: {accuracy:.2f}")
    st.write(f"Precisión (Precision): {precision:.2f}")
    st.write(f"Exhaustividad (Recall): {recall:.2f}")
    st.write(f"F1-Score: {f1:.2f}")
    st.write(f"ROC-AUC: {roc_auc:.2f}")
    
    # Curva ROC-AUC
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Curva ROC-AUC')
    ax.legend(loc='lower right')
    st.pyplot(fig)
    
    # Ingreso de nuevos datos para hacer una predicción
    st.subheader("Haz una predicción para un nuevo pasajero")
    pclass = st.selectbox("Clase del pasajero (1, 2, 3)", [1, 2, 3])
    sex = st.radio("Sexo", ['Mujer', 'Hombre'])
    age = st.number_input("Edad", min_value=0)
    sibsp = st.number_input("Hermanos o esposos a bordo", min_value=0)
    parch = st.number_input("Padres o hijos a bordo", min_value=0)
    fare = st.number_input("Tarifa", min_value=0.0)
    
    # Convertir sexo a 0 o 1
    sex = 0 if sex == 'Mujer' else 1
    
    if st.button('Predecir'):
        pred = model.predict([[pclass, sex, age, sibsp, parch, fare]])
        if pred == 1:
            st.write("La persona sobrevivió.")
        else:
            st.write("La persona no sobrevivió.")
    
    # Preparar los resultados para la exportación
    survivors = pd.DataFrame(y_pred, columns=['Survived'])
    survivors.insert(0, 'PassengerId', test_data['PassengerId'], True)
    survivors.insert(1, 'Name', test_data['Name'], True)
    
    st.subheader("Resultados de la Predicción")
    st.write(survivors)  # Muestra todo el DataFrame con los resultados de la predicción
    
    # Descargar el archivo de resultados
    survivors_file = survivors.to_csv(index=False)
    st.download_button("Descargar Resultados", survivors_file, file_name='titanic_predictions.csv')

else:
    st.warning("Por favor, sube los archivos 'train.csv' y 'test.csv'.")

# Sección de ayuda en el sidebar
with st.sidebar:
    # Botón para descargar el archivo Titanic.pdf
    with open("Titanic.pdf", "rb") as f:
        st.download_button("Descargar Titanic.pdf", f, file_name="Titanic.pdf")

    # Ayuda
    st.title("Ayuda")
    st.write("""
    Esta aplicación predice la supervivencia de los pasajeros del Titanic usando un modelo de árbol de decisión.

    **Pasos del flujo de trabajo:**

    1. **Subir Archivos:** Suba los archivos CSV de entrenamiento (`train.csv`) y prueba (`test.csv`).
    2. **Exploración de Datos:** Visualiza diferentes distribuciones de los datos, como la distribución de los sobrevivientes por género, clase y cantidad de hijos.
    3. **Preprocesamiento de Datos:** Se realiza el reemplazo de valores categóricos y el manejo de valores nulos en la columna `Age`.
    4. **Entrenamiento del Modelo:** Se entrena un modelo de árbol de decisión utilizando las características `Pclass`, `Sex`, `Age`, `SibSp`, `Parch` y `Fare`.
    5. **Evaluación del Modelo:** Se presentan las métricas de evaluación del modelo: precisión, precisión, exhaustividad, F1-Score y ROC-AUC.
    6. **Predicción en Nuevos Datos:** Puede ingresar datos de nuevos pasajeros para predecir si sobrevivieron o no.
    7. **Descargar Resultados:** Los resultados de la predicción se pueden descargar en formato CSV.

    """)


with st.sidebar:
    st.markdown("## © Copyrith")
    st.markdown("**Javier Horacio Pérez Ricárdez**")
    st.markdown("Aprendizaje máquina I")
    st.markdown("**Catedrático: DR. Félix Orlando Martínez**")
