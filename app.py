import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

gamma = 0.75
alpha = 0.9
Q = np.array(np.zeros([20,20]))

def_de_estados = {'A': 0,'B': 1,'C': 2,'D': 3,'E': 4,'F': 5,'G': 6,'H': 7,'I': 8,'J': 9,
                'K': 10,'L': 11,'M': 12,'N': 13,'Ñ': 14,'O': 15,'P': 16,'Q': 17,'R': 18,'S': 19}

def asignar_recompensa(R):
    with st.sidebar:
        try:
            st.header("Recompensas")
            st.write("Selecciona las coordenadas para la recompensa mayor:")
            coordenada_10 = st.selectbox("Coordenadas para la recompensa mayor", list(def_de_estados.keys()))
            R[def_de_estados[coordenada_10], def_de_estados[coordenada_10]] = 10
            
            respuesta = st.radio("¿Deseas ingresar una recompensa de intermedia?", ('Sí', 'No'))
            if respuesta == 'Sí':
                    coordenadas_5 = st.selectbox("Coordenadas para la recompensa intermedia", list(def_de_estados.keys()))
                    fila_5, columna_5 = map(int, coordenadas_5.split(','))
                    R[fila_5, columna_5] = 5
        except:
            pass
def ingresar_inicio():
    with st.sidebar:
        try:
            st.header("Punto de inicio")
            st.write("Selecciona las coordenadas para el punto de inicio:")
            inicio = st.selectbox("Coordenadas para el punto de inicio", list(def_de_estados.keys()))
            return def_de_estados[inicio]
        except:
            pass
def encontrar_ruta(Q, inicio):
    ruta = []  
    estado_actual = inicio
    visitados = set()  

    while estado_actual not in visitados:
        visitados.add(estado_actual)
        estado_siguiente = np.argmax(Q[estado_actual, :])
        ruta.append(str(estado_actual))
        estado_actual = estado_siguiente

    return ruta

def main():
    st.subheader("Optimización de Rutas con Q-Learning")
    
    R = np.array ([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]])
    inicio = ingresar_inicio()
    asignar_recompensa(R)
    

    for i in range(1000):
        estado_actual, estado_siguiente = ambiente(R)
        TD = R[estado_actual, estado_siguiente] + gamma*Q[estado_siguiente, np.argmax(Q[estado_siguiente,])] - Q[estado_actual, estado_siguiente]
        Q[estado_actual, estado_siguiente] = Q[estado_actual, estado_siguiente] + alpha*TD

    ruta = encontrar_ruta(Q, inicio)
    
    st.image("grafo.jpeg")
    st.markdown("---")
    st.write("\nEl punto de inicio es:", inicio)
    st.write("\nEl punto final es:", ruta[-1])
    st.write("\nLa ruta que encontró el agente es:", " -> ".join(ruta))

def ambiente(R):
    estado_actual = np.random.randint(0,20)
    accion_realizable = []
    for j in range(20):
        if R[estado_actual, j] > 0:
            accion_realizable.append(j)
    estado_siguiente = np.random.choice(accion_realizable)
    return estado_actual, estado_siguiente

if __name__ == "__main__":
    main()
