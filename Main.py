# Suftiwa 2.4
# Mini red neuronal conversacional en español
# Compatible con Python / Pydroid 3
# 100% offline

import os
import json
import random
import math

MODEL_FILE = "suftiwa_2_4.json"

# ============================================================
# DATOS DE ENTRENAMIENTO
# ============================================================

ENTRENAMIENTO_1 = [
    ("hola", "hola"),
    ("hola suftiwa", "hola"),
    ("buenas", "buenas"),
    ("buenos dias", "buenos dias"),
    ("buenas tardes", "buenas tardes"),
    ("buenas noches", "buenas noches"),
    ("como estas", "estoy bien"),
    ("como te llamas", "me llamo suftiwa"),
    ("quien eres", "soy suftiwa 2.4"),
    ("que eres", "soy una red neuronal"),
    ("eres una ia", "si, soy una inteligencia artificial"),
    ("adios", "adios"),
    ("hasta luego", "hasta luego"),
]

ENTRENAMIENTO_2 = [
    ("que puedes hacer", "puedo conversar contigo"),
    ("puedes hablar español", "si, hablo español"),
    ("hablas español", "si, hablo español"),
    ("cual es tu nombre", "mi nombre es suftiwa"),
    ("como te llamas", "me llamo suftiwa"),
    ("quien te creo", "fui creada mediante entrenamiento neuronal"),
    ("eres inteligente", "puedo aprender patrones mediante entrenamiento"),
    ("puedes aprender", "si, puedo entrenarme con nuevos datos"),
    ("que es una red neuronal", "es un modelo que aprende patrones de los datos"),
    ("que es una ia", "es un sistema capaz de realizar tareas que requieren inteligencia"),
]

ENTRENAMIENTO_3 = [
    ("que hora es", "no puedo ver la hora actual"),
    ("cuentame algo", "los pulpos tienen tres corazones"),
    ("dime un dato", "la luz del sol tarda unos ocho minutos en llegar a la tierra"),
    ("que es españa", "es un pais de europa"),
    ("que es honduras", "es un pais de centroamerica"),
    ("que es mexico", "es un pais de norteamerica"),
    ("me gusta la tecnologia", "la tecnologia es muy interesante"),
    ("me gustan los videojuegos", "los videojuegos pueden ser muy divertidos"),
    ("gracias", "de nada"),
    ("te quiero", "gracias por decirlo"),
]

DATOS = ENTRENAMIENTO_1 + ENTRENAMIENTO_2 + ENTRENAMIENTO_3


# ============================================================
# UTILIDADES
# ============================================================

def softmax(x):
    mayor = max(x)
    exp = [math.exp(v - mayor) for v in x]
    total = sum(exp)
    return [v / total for v in exp]


def limpiar(texto):
    texto = texto.lower().strip()

    reemplazos = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n"
    }

    for a, b in reemplazos.items():
        texto = texto.replace(a, b)

    return texto


# ============================================================
# TOKENIZADOR
# ============================================================

def construir_vocabulario(datos):
    caracteres = set()

    for pregunta, respuesta in datos:
        caracteres.update(limpiar(pregunta))
        caracteres.update(limpiar(respuesta))

    caracteres.update(["<", ">", " "])

    caracteres = sorted(caracteres)

    char_to_id = {
        c: i for i, c in enumerate(caracteres)
    }

    id_to_char = {
        i: c for i, c in enumerate(caracteres)
    }

    return char_to_id, id_to_char


# ============================================================
# RED NEURONAL
# ============================================================

class SuftiwaNN:

    def __init__(self, entrada, oculta, salida):

        self.entrada = entrada
        self.oculta = oculta
        self.salida = salida

        random.seed(24)

        self.W1 = [
            [
                random.uniform(-0.5, 0.5)
                for _ in range(oculta)
            ]
            for _ in range(entrada)
        ]

        self.W2 = [
            [
                random.uniform(-0.5, 0.5)
                for _ in range(salida)
            ]
            for _ in range(oculta)
        ]

        self.b1 = [0.0] * oculta
        self.b2 = [0.0] * salida


    def forward(self, x):

        ocultas = []

        for j in range(self.oculta):

            suma = self.b1[j]

            for i in range(self.entrada):
                suma += x[i] * self.W1[i][j]

            ocultas.append(math.tanh(suma))

        salida = []

        for k in range(self.salida):

            suma = self.b2[k]

            for j in range(self.oculta):
                suma += ocultas[j] * self.W2[j][k]

            salida.append(suma)

        return ocultas, softmax(salida)


    def entrenar(self, x, objetivo, lr=0.03):

        ocultas, pred = self.forward(x)

        # Error de salida
        error_salida = [
            pred[i] - objetivo[i]
            for i in range(self.salida)
        ]

        # Error de capa oculta
        error_oculta = [0.0] * self.oculta

        for j in range(self.oculta):

            suma = 0

            for k in range(self.salida):
                suma += error_salida[k] * self.W2[j][k]

            error_oculta[j] = (
                suma * (1 - ocultas[j] ** 2)
            )

        # Actualizar W2
        for j in range(self.oculta):
            for k in range(self.salida):
                self.W2[j][k] -= (
                    lr *
                    error_salida[k] *
                    ocultas[j]
                )

        # Actualizar bias salida
        for k in range(self.salida):
            self.b2[k] -= lr * error_salida[k]

        # Actualizar W1
        for i in range(self.entrada):
            for j in range(self.oculta):
                self.W1[i][j] -= (
                    lr *
                    error_oculta[j] *
                    x[i]
                )

        # Actualizar bias oculta
        for j in range(self.oculta):
            self.b1[j] -= lr * error_oculta[j]


# ============================================================
# ENTRENAMIENTO
# ============================================================

def entrenar_modelo():

    print("\n================================")
    print("       SUFTIWA 2.4")
    print("================================")
    print("Entrenamiento neuronal")
    print()

    datos = DATOS

    char_to_id, id_to_char = construir_vocabulario(datos)

    tamano = len(char_to_id)

    red = SuftiwaNN(
        tamano,
        64,
        tamano
    )

    # Entrenamientos separados
    entrenamientos = [
        ("Entrenamiento 1: básico", ENTRENAMIENTO_1),
        ("Entrenamiento 2: conversación", ENTRENAMIENTO_2),
        ("Entrenamiento 3: conocimiento", ENTRENAMIENTO_3)
    ]

    for nombre, conjunto in entrenamientos:

        print("\n" + nombre)

        for epoca in range(300):

            for pregunta, respuesta in conjunto:

                pregunta = limpiar(pregunta)
                respuesta = limpiar(respuesta)

                entrada = [0.0] * tamano
                objetivo = [0.0] * tamano

                for c in pregunta:
                    if c in char_to_id:
                        entrada[char_to_id[c]] += 1

                for c in respuesta:
                    if c in char_to_id:
                        objetivo[char_to_id[c]] += 1

                suma = sum(objetivo)

                if suma:
                    objetivo = [
                        v / suma
                        for v in objetivo
                    ]

                red.entrenar(
                    entrada,
                    objetivo,
                    lr=0.025
                )

            if epoca % 50 == 0:
                print("Época:", epoca)

    modelo = {
        "nombre": "Suftiwa 2.4",
        "version": "2.4",
        "entrada": red.entrada,
        "oculta": red.oculta,
        "salida": red.salida,
        "W1": red.W1,
        "W2": red.W2,
        "b1": red.b1,
        "b2": red.b2,
        "char_to_id": char_to_id,
        "id_to_char": id_to_char
    }

    with open(MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump(modelo, f)

    print("\nEntrenamiento terminado.")
    print("Modelo guardado como:", MODEL_FILE)


# ============================================================
# CARGAR MODELO
# ============================================================

def cargar_modelo():

    if not os.path.exists(MODEL_FILE):
        print("No existe un modelo entrenado.")
        print("Entrena Suftiwa primero.")
        return None

    with open(MODEL_FILE, "r", encoding="utf-8") as f:
        modelo = json.load(f)

    red = SuftiwaNN(
        modelo["entrada"],
        modelo["oculta"],
        modelo["salida"]
    )

    red.W1 = modelo["W1"]
    red.W2 = modelo["W2"]
    red.b1 = modelo["b1"]
    red.b2 = modelo["b2"]

    return red, modelo


# ============================================================
# RESPUESTA
# ============================================================

def responder(red, modelo, texto):

    texto = limpiar(texto)

    char_to_id = modelo["char_to_id"]

    entrada = [0.0] * red.entrada

    for c in texto:

        if c in char_to_id:
            entrada[char_to_id[c]] += 1

    ocultas, probabilidades = red.forward(entrada)

    mejores = sorted(
        range(len(probabilidades)),
        key=lambda i: probabilidades[i],
        reverse=True
    )

    indice = mejores[0]

    confianza = probabilidades[indice]

    id_to_char = {
        int(k): v
        for k, v in modelo["id_to_char"].items()
    }

    # Buscar una respuesta conocida
    mejor_pregunta = None
    mejor_respuesta = None
    mejor_similitud = 0

    for pregunta, respuesta in DATOS:

        p = limpiar(pregunta)

        conjunto_a = set(texto.split())
        conjunto_b = set(p.split())

        if conjunto_a or conjunto_b:

            interseccion = len(conjunto_a & conjunto_b)
            union = len(conjunto_a | conjunto_b)

            similitud = interseccion / union if union else 0

            if similitud > mejor_similitud:
                mejor_similitud = similitud
                mejor_pregunta = pregunta
                mejor_respuesta = respuesta

    if mejor_respuesta and mejor_similitud >= 0.25:
        return mejor_respuesta

    respuestas_genericas = [
        "interesante",
        "entiendo",
        "puedes explicarme un poco mas",
        "estoy aprendiendo sobre eso",
        "no conozco esa respuesta todavia"
    ]

    return random.choice(respuestas_genericas)


# ============================================================
# CHAT
# ============================================================

def chat():

    resultado = cargar_modelo()

    if resultado is None:
        return

    red, modelo = resultado

    print("\n================================")
    print("       SUFTIWA 2.4 CHAT")
    print("================================")
    print("Escribe 'salir' para terminar.")
    print()

    while True:

        usuario = input("Tu: ")

        if usuario.lower().strip() == "salir":
            print("Suftiwa: hasta luego.")
            break

        respuesta = responder(
            red,
            modelo,
            usuario
        )

        print("Suftiwa:", respuesta)


# ============================================================
# MENU
# ============================================================

while True:

    print("\n")
    print("╔══════════════════════════════╗")
    print("║       SUFTIWA 2.4            ║")
    print("║   Red neuronal en español    ║")
    print("╠══════════════════════════════╣")
    print("║ 1. Entrenar                  ║")
    print("║ 2. Hablar con Suftiwa        ║")
    print("║ 3. Salir                     ║")
    print("╚══════════════════════════════╝")

    opcion = input("\nSelecciona: ")

    if opcion == "1":
        entrenar_modelo()

    elif opcion == "2":
        chat()

    elif opcion == "3":
        print("Suftiwa cerrada.")
        break

    else:
        print("Opción inválida.")