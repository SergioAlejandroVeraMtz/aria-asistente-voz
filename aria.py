"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        AGENTE ARIA — Sistema Conversacional Basado en Voz                  ║
║        Reconocimiento · Comprensión · Diálogo · Generación · Síntesis      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Equipo Papois — ESCOM / IPN — Procesamiento de Lenguaje Natural 2026      ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Vera Martínez Sergio Alejandro                     ║
║  Santiago de Luis Mauricio                     ║
║  Luis Fernando Martínez González               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Arquitectura de 5 módulos:                                                 ║
║    ASR  Reconocimiento Automático de Habla (audio → texto)                  ║
║    NLU  Comprensión de Lenguaje Natural    (texto → intención + entidades)  ║
║    DM   Gestor de Diálogo MEF             (estado + decisión)               ║
║    NLG  Generación de Lenguaje Natural    (acción → respuesta)              ║
║    TTS  Síntesis de Voz                   (texto → audio)                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Dependencias opcionales (todas con fallback sin ellas):                    ║
║    pip install gtts SpeechRecognition pyaudio                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Uso desde terminal:                                                         ║
║    python agente_aria_entrega.py              → Menú interactivo            ║
║    python agente_aria_entrega.py --eval       → Evaluación NLU completa     ║
║    python agente_aria_entrega.py --mfcc       → Demo coeficientes MFCC      ║
║    python agente_aria_entrega.py --voz        → Modo micrófono              ║
║    python agente_aria_entrega.py --sin-tts    → Sin síntesis de voz         ║
║    python agente_aria_entrega.py --historial archivo.json                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Importaciones de la biblioteca estándar ───────────────────────────────────
import os
import re
import sys
import math
import json
import time
import random
import datetime
import tempfile
import argparse
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

# ── Importaciones opcionales ──────────────────────────────────────────────────
try:
    from gtts import gTTS
    TTS_OK = True
except ImportError:
    TTS_OK = False

def _intentar_instalar_asr() -> bool:
    """
    Intenta instalar SpeechRecognition y pyaudio automáticamente
    si no están presentes. Retorna True si la instalación fue exitosa.
    """
    import subprocess
    print("  Instalando SpeechRecognition y pyaudio…", flush=True)
    # En macOS instalar portaudio primero si brew está disponible
    if sys.platform == "darwin":
        subprocess.run(["brew", "install", "portaudio"],
                       capture_output=True)
    resultado = subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "SpeechRecognition", "pyaudio", "--quiet"],
        capture_output=True, text=True
    )
    return resultado.returncode == 0

def _check_asr() -> bool:
    """Verifica si SpeechRecognition está disponible en el entorno actual."""
    try:
        import speech_recognition  # noqa: F401
        return True
    except ImportError:
        return False

ASR_OK = _check_asr()


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 0 — UTILIDADES DE TERMINAL
# ══════════════════════════════════════════════════════════════════════════════

class C:
    """
    Códigos de color ANSI para salida en terminal.
    Se desactivan automáticamente si la salida no es un TTY
    (p. ej. redireccionamiento a archivo).
    """
    CYAN  = '\033[96m';  BLUE  = '\033[94m'
    RED   = '\033[91m';  GREEN = '\033[92m'
    GOLD  = '\033[93m';  GRAY  = '\033[90m'
    BOLD  = '\033[1m';   DIM   = '\033[2m'
    RESET = '\033[0m'

    if not sys.stdout.isatty():
        CYAN=BLUE=RED=GREEN=GOLD=GRAY=BOLD=DIM=RESET=''


def cprint(color: str, texto: str) -> None:
    """Imprime texto con color ANSI."""
    print(f"{color}{texto}{C.RESET}")


def box(titulo: str, color: str = C.CYAN, ancho: int = 64) -> None:
    """Imprime un título enmarcado con una línea decorativa."""
    print(f"\n{color}{'─' * ancho}{C.RESET}")
    print(f"{color}  {titulo}{C.RESET}")
    print(f"{color}{'─' * ancho}{C.RESET}")


def barra(actual: int, total: int, ancho: int = 28) -> str:
    """Genera una barra de progreso ASCII proporcional."""
    pct   = actual / max(total, 1)
    lleno = int(pct * ancho)
    return f"[{'█' * lleno}{'░' * (ancho - lleno)}] {pct * 100:.0f}%"


def normalizar(texto: str) -> str:
    """
    Normalización de texto para NLU:
      - Convierte a minúsculas
      - Elimina caracteres de control (mantiene acentos del español)
      - Colapsa espacios múltiples
    """
    texto = texto.lower().strip()
    texto = ''.join(c for c in texto
                    if unicodedata.category(c) not in ('Cc', 'Cf'))
    texto = re.sub(r'\s+', ' ', texto)
    return texto


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 1 — ASR: Reconocimiento Automático de Habla
# ══════════════════════════════════════════════════════════════════════════════

class ASR:
    """
    Módulo ASR — Reconocimiento Automático de Habla.

    Convierte señal de audio a texto transcrito.
    Implementa además el cálculo didáctico de MFCC desde cero.

    Backends disponibles:
      'google'  — Google Speech-to-Text (requiere internet)
      'sphinx'  — CMU Sphinx (offline, requiere pocketsphinx)
      'sim'     — Simulador por teclado (fallback siempre disponible)

    Pipeline de extracción de características MFCC:
      Señal → Pre-énfasis → Ventana Hamming → FFT → Banco Mel → Log → DCT
    """

    def __init__(self, idioma: str = "es-MX", backend: str = "auto"):
        """
        Inicializa el módulo ASR.

        Args:
            idioma:  Código BCP-47 del idioma (p. ej. 'es-MX', 'es-ES').
            backend: 'auto' selecciona Google si SpeechRecognition está instalado,
                     'sim' fuerza el simulador por teclado.
        """
        self.idioma  = idioma
        self.backend = backend
        self._rec: Optional[object] = None
        self._sr = None  # referencia local al módulo

        try:
            import speech_recognition as _sr
            self._sr     = _sr
            self._rec    = _sr.Recognizer()
            self._rec.dynamic_energy_threshold = True
            self._rec.pause_threshold          = 0.8
            self.backend = "google" if backend == "auto" else backend
        except ImportError:
            self.backend = "sim"

        print(f"  \033[2m[ASR] backend={self.backend} · idioma={self.idioma}\033[0m")

    # ─── Captura y transcripción ──────────────────────────────────────────────

    def escuchar(self, timeout: int = 6) -> str:
        """
        Graba audio del micrófono y retorna el texto transcrito.

        Si el backend es 'sim' o SpeechRecognition no está disponible,
        devuelve una frase simulada aleatoria del corpus de prueba.

        Args:
            timeout: Segundos máximos de espera por audio.

        Returns:
            Texto transcrito (puede ser cadena vacía si no se detecta audio).
        """
        if self.backend == "sim" or self._sr is None:
            return self._simular()

        try:
            with self._sr.Microphone() as fuente:
                print("  \033[96m[ASR] Calibrando ruido ambiental…\033[0m", end="\r")
                self._rec.adjust_for_ambient_noise(fuente, duration=0.8)
                print("  \033[92m[ASR] Escuchando… (habla ahora)   \033[0m")
                audio = self._rec.listen(fuente, timeout=timeout,
                                         phrase_time_limit=15)
            return self._transcribir(audio)

        except self._sr.WaitTimeoutError:
            print("  \033[90m[ASR] Sin audio en el tiempo límite.\033[0m")
            return ""
        except Exception as exc:
            print(f"  \033[91m[ASR] Error de micrófono: {exc}\033[0m")
            return self._simular()

    def _transcribir(self, audio) -> str:
        """Envía el audio al backend configurado y retorna el texto."""
        try:
            texto = self._rec.recognize_google(audio, language=self.idioma)
            print(f"  {C.DIM}[ASR] Transcrito: «{texto}»{C.RESET}")
            return texto
        except self._sr.UnknownValueError:
            print(f"  {C.GRAY}[ASR] Audio no reconocido.{C.RESET}")
            return ""
        except self._sr.RequestError as exc:
            print(f"  {C.RED}[ASR] Servicio no disponible: {exc}{C.RESET}")
            return ""

    @staticmethod
    def _simular() -> str:
        """Retorna una frase aleatoria para demos sin micrófono."""
        frases = [
            "hola buenos días", "qué hora es", "cuéntame un chiste",
            "qué puedes hacer", "cuál es tu nombre", "cuánto es 5 más 3",
            "gracias", "hasta luego",
        ]
        sel = random.choice(frases)
        print(f"  {C.DIM}[ASR-SIM] Frase simulada: «{sel}»{C.RESET}")
        return sel

    # ─── Cálculo didáctico de MFCC ────────────────────────────────────────────

    @staticmethod
    def calcular_mfcc(señal: list,
                      fs: int = 16_000,
                      n_filtros: int = 26,
                      n_coef: int = 13) -> list:
        """
        Calcula los Coeficientes Cepstrales en las Frecuencias de Mel (MFCC).
        Implementación didáctica paso a paso sin dependencias externas.

        Pasos del pipeline:
          1. Pre-énfasis:     y[n] = x[n] - 0.97·x[n-1]
             Compensa la caída de ~20 dB/década en la señal de voz.
          2. Ventana Hamming: w[n] = 0.54 - 0.46·cos(2πn/(N-1))
             Reduce el efecto de borde al segmentar la señal.
          3. FFT (DFT manual O(N²), M=256 puntos):
             Transforma al dominio de frecuencia.
          4. Banco de M filtros triangulares en escala Mel:
             Mel = 2595·log10(1 + f/700)
             Imita la resolución auditiva humana (logarítmica).
          5. Logaritmo de las energías por filtro.
          6. DCT para n_coef coeficientes cepstrales finales.

        Args:
            señal:     Lista de muestras numéricas.
            fs:        Frecuencia de muestreo en Hz.
            n_filtros: Número de filtros Mel (por defecto 26).
            n_coef:    Número de coeficientes MFCC de salida (por defecto 13).

        Returns:
            Lista de n_coef coeficientes MFCC redondeados a 6 decimales.
        """
        if len(señal) < 2:
            return [0.0] * n_coef

        # ── Paso 1: Pre-énfasis ──────────────────────────────────────────────
        alfa = 0.97
        pe   = [señal[0]] + [señal[i] - alfa * señal[i - 1]
                              for i in range(1, len(señal))]

        # ── Paso 2: Ventana de Hamming ───────────────────────────────────────
        N   = len(pe)
        win = [0.54 - 0.46 * math.cos(2 * math.pi * n / (N - 1))
               for n in range(N)]
        sv  = [pe[i] * win[i] for i in range(N)]

        # ── Paso 3: DFT manual (O(N²), M=256 puntos) ────────────────────────
        M   = min(N, 256)
        esp = []
        for k in range(M // 2):
            r = sum(sv[n] * math.cos(2 * math.pi * k * n / M) for n in range(M))
            i = sum(sv[n] * math.sin(2 * math.pi * k * n / M) for n in range(M))
            esp.append(math.sqrt(r * r + i * i))

        # ── Paso 4: Banco de filtros triangulares en escala Mel ──────────────
        hz2mel = lambda h: 2595 * math.log10(1 + h / 700)
        mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)

        m_min, m_max = hz2mel(0), hz2mel(fs / 2)
        mels = [m_min + i * (m_max - m_min) / (n_filtros + 1)
                for i in range(n_filtros + 2)]
        bins = [int((M // 2 + 1) * mel2hz(m) / (fs / 2)) for m in mels]

        energias = []
        for m in range(1, n_filtros + 1):
            e = 0.0
            # Rampa ascendente
            for k in range(bins[m - 1], min(bins[m], len(esp))):
                e += ((k - bins[m - 1]) / max(bins[m] - bins[m - 1], 1)) * esp[k]
            # Rampa descendente
            for k in range(bins[m], min(bins[m + 1], len(esp))):
                e += ((bins[m + 1] - k) / max(bins[m + 1] - bins[m], 1)) * esp[k]
            energias.append(math.log(max(e, 1e-10)))

        # ── Paso 5: DCT → n_coef coeficientes ───────────────────────────────
        mfcc = [
            sum(energias[m] * math.cos(math.pi * n * (m + 0.5) / n_filtros)
                for m in range(n_filtros))
            for n in range(n_coef)
        ]
        return [round(c, 6) for c in mfcc]

    @staticmethod
    def energia_rms(muestras: list) -> float:
        """
        Calcula la Energía RMS (Root Mean Square) del frame.
        Proxy de la intensidad sonora percibida.
        """
        return math.sqrt(sum(x * x for x in muestras) / max(len(muestras), 1))

    @staticmethod
    def tasa_cruces_cero(muestras: list) -> float:
        """
        Calcula la Tasa de Cruces por Cero (ZCR).
        Indicador de la presencia de consonantes fricativas o sonidos sordos.
        """
        return (sum(1 for i in range(1, len(muestras))
                    if muestras[i] * muestras[i - 1] < 0)
                / max(len(muestras) - 1, 1))


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 2 — NLU: Comprensión de Lenguaje Natural
# ══════════════════════════════════════════════════════════════════════════════

class NLU:
    """
    Módulo NLU — Comprensión de Lenguaje Natural.

    Detecta la intención del usuario en el texto transcrito y extrae
    entidades relevantes (números, horas, fechas, operaciones).

    Algoritmo de detección de intenciones (3 pasos):
      1. Coincidencia exacta de frases clave → confianza 1.0  (O(k·P))
      2. Vectorización TF-IDF + similitud coseno              (O(V·C))
      3. Umbral de rechazo: score < 0.15 → 'desconocido'

    Corpus: 14 intenciones con ~8 ejemplos cada una = ~112 frases de entrenamiento.
    """

    # ── Corpus de intenciones y ejemplos ─────────────────────────────────────
    INTENCIONES: dict = {
        "saludo": [
            "hola", "buenos días", "buenas tardes", "buenas noches",
            "qué tal", "hey", "buenas", "saludos", "hola qué tal",
            "cómo estás", "qué onda", "buen día",
        ],
        "despedida": [
            "adiós", "hasta luego", "chao", "nos vemos", "bye",
            "hasta pronto", "me voy", "hasta mañana", "hasta la vista",
            "me despido", "chau", "buenas noches",
        ],
        "nombre": [
            "cómo te llamas", "cuál es tu nombre", "quién eres",
            "nombre", "cómo te llaman", "tu nombre", "dime tu nombre",
            "qué nombre tienes", "cómo debo llamarte",
        ],
        "ayuda": [
            "ayuda", "ayúdame", "qué puedes hacer", "qué sabes hacer",
            "comandos", "opciones", "help", "cómo funciona",
            "para qué sirves", "qué haces", "instrucciones",
        ],
        "hora": [
            "qué hora es", "hora", "dime la hora", "qué tiempo es",
            "hora actual", "me dices la hora", "qué horas son",
            "dame la hora", "dime qué hora es",
        ],
        "fecha": [
            "qué día es", "fecha", "cuál es la fecha", "hoy qué día es",
            "día de hoy", "qué fecha es hoy", "dime la fecha",
            "cuál es la fecha de hoy", "en qué fecha estamos",
        ],
        "chiste": [
            "chiste", "cuéntame algo gracioso", "hazme reír", "broma",
            "cuéntame un chiste", "algo chistoso", "dime un chiste",
            "cuéntame una broma", "algo divertido",
        ],
        "clima": [
            "clima", "temperatura", "llueve", "hace frío", "hace calor",
            "tiempo", "cómo está el clima", "qué tiempo hace",
            "va a llover", "pronóstico del tiempo",
        ],
        "afirmacion": [
            "sí", "correcto", "exacto", "así es", "claro", "ok",
            "de acuerdo", "entendido", "efectivamente", "por supuesto",
            "afirmativo", "desde luego",
        ],
        "negacion": [
            "no", "negativo", "incorrecto", "para nada",
            "de ninguna manera", "tampoco", "no gracias",
            "en absoluto", "claro que no",
        ],
        "gracias": [
            "gracias", "muchas gracias", "te lo agradezco",
            "mil gracias", "te agradezco", "muy amable",
            "gracias por todo",
        ],
        "insulto": [
            "tonto", "idiota", "estúpido", "inútil", "malo",
            "eres una basura", "eres pésimo", "no sirves",
        ],
        "matematica": [
            "cuánto es", "calcula", "suma", "resta", "multiplica",
            "divide", "cuánto da", "resultado de", "cuál es el resultado",
        ],
        "presentacion": [
            "preséntate", "dime quién eres", "cuéntame sobre ti",
            "qué eres", "quién te creó", "quién te hizo",
            "de dónde eres", "quiénes te desarrollaron",
        ],
        "desconocido": [],
    }

    CHISTES: list = [
        "¿Por qué los programadores confunden Halloween con Navidad?\n"
        "  Porque OCT 31 = DEC 25.",
        "¿Cómo se despide un informático?\n"
        "  Hasta el bit te vea.",
        "¿Por qué el programador llevó una escalera al trabajo?\n"
        "  Porque su jefe dijo que quería subir el nivel del código.",
        "¿Qué le dijo el 0 al 8?\n"
        "  Qué bonito cinturón.",
        "¿Cuántos programadores se necesitan para cambiar un foco?\n"
        "  Ninguno, eso es un problema de hardware.",
        "Un bucle infinito entra a un bar y pide una cerveza.\n"
        "  El camarero le pregunta: ¿la misma de siempre?\n"
        "  El bucle responde: sí, la misma de siempre.",
        "¿Cuál es el colmo de un programador?\n"
        "  Que sus hijos le llamen papá solo en modo debug.",
    ]

    def __init__(self):
        """Construye el corpus y pre-calcula los pesos IDF."""
        self._corpus: list = []   # [(intención, texto_ejemplo), …]
        self._idf:    dict = {}   # {término: peso_idf}

        for intent, frases in self.INTENCIONES.items():
            for frase in frases:
                self._corpus.append((intent, frase.lower()))

        self._calcular_idf()
        print(f"  {C.DIM}[NLU] Corpus: {len(self._corpus)} ejemplos · "
              f"{len(self.INTENCIONES)} intenciones{C.RESET}")

    # ─── Funciones de vectorización TF-IDF ───────────────────────────────────

    def _tokenizar(self, texto: str) -> list:
        """Tokeniza el texto en palabras alfanuméricas en minúsculas."""
        return re.findall(r'\b\w+\b', texto.lower())

    def _tf(self, tokens: list) -> dict:
        """
        Calcula la Frecuencia de Término (TF) normalizada.
        TF(t, d) = frecuencia(t en d) / total_términos(d)
        """
        conteo = Counter(tokens)
        total  = max(len(tokens), 1)
        return {t: v / total for t, v in conteo.items()}

    def _calcular_idf(self) -> None:
        """
        Calcula la Frecuencia Inversa de Documento (IDF) para todo el corpus.
        IDF(t, D) = log(|D| / (1 + |{d ∈ D : t ∈ d}|))
        """
        N  = max(len(self._corpus), 1)
        df: dict = defaultdict(int)
        for _, doc in self._corpus:
            for term in set(self._tokenizar(doc)):
                df[term] += 1
        self._idf = {t: math.log(N / (1 + v)) for t, v in df.items()}

    def _vectorizar(self, texto: str) -> dict:
        """
        Convierte texto a vector TF-IDF disperso.
        vec[t] = TF(t, texto) × IDF(t, corpus)
        """
        tokens = self._tokenizar(texto)
        tf     = self._tf(tokens)
        return {k: tf[k] * self._idf.get(k, 0.0) for k in tf}

    def _coseno(self, va: dict, vb: dict) -> float:
        """
        Calcula la similitud coseno entre dos vectores dispersos.
        sim(u, d) = (u · d) / (‖u‖ · ‖d‖)
        """
        comunes = set(va) & set(vb)
        dot     = sum(va[k] * vb[k] for k in comunes)
        norma_a = math.sqrt(sum(x * x for x in va.values()))
        norma_b = math.sqrt(sum(x * x for x in vb.values()))
        return dot / (norma_a * norma_b + 1e-9)

    # ─── Detección de intención ───────────────────────────────────────────────

    def detectar(self, texto: str) -> tuple:
        """
        Detecta la intención del texto de entrada.

        Algoritmo de 3 pasos:
          Paso 1 — Coincidencia exacta:
            Si alguna frase de entrenamiento aparece en el texto,
            retorna esa intención con confianza = 1.0.
          Paso 2 — TF-IDF + coseno:
            Vectoriza el texto y compara contra todo el corpus.
            Retorna la intención con mayor similitud coseno.
          Paso 3 — Umbral de rechazo:
            Si el mejor score < 0.15, retorna 'desconocido'.

        Args:
            texto: Enunciado del usuario (sin preprocesar).

        Returns:
            Tupla (intención: str, confianza: float).
        """
        n = normalizar(texto)

        # Paso 1: coincidencia exacta → confianza máxima
        for intent, patrones in self.INTENCIONES.items():
            for patron in patrones:
                if patron in n:
                    return intent, 1.0

        # Paso 2: TF-IDF + similitud coseno
        vec    = self._vectorizar(n)
        mejor  = "desconocido"
        score  = -1.0
        for intent, doc in self._corpus:
            s = self._coseno(vec, self._vectorizar(doc))
            if s > score:
                score, mejor = s, intent

        # Paso 3: umbral mínimo de confianza
        if score < 0.15:
            return "desconocido", round(score, 4)
        return mejor, round(score, 4)

    # ─── Extracción de entidades ──────────────────────────────────────────────

    def extraer_entidades(self, texto: str) -> dict:
        """
        Extrae entidades nombradas del texto mediante expresiones regulares.

        Tipos de entidades detectadas:
          - numeros:           Lista de valores numéricos encontrados.
          - hora_mencionada:   Hora en formato HH:MM (opcional: am/pm).
          - fecha_mencionada:  Fecha en formato DD/MM/AAAA o DD-MM-AAAA.
          - operacion:         Operación aritmética simple (a OP b).

        Returns:
            Diccionario con las entidades detectadas (vacío si no hay).
        """
        entidades: dict = {}

        # Números
        nums = re.findall(r'\b\d+(?:\.\d+)?\b', texto)
        if nums:
            entidades["numeros"] = [float(n) for n in nums]

        # Hora HH:MM
        hora = re.search(r'\b(\d{1,2}):(\d{2})(?:\s*[ap]m)?\b', texto, re.I)
        if hora:
            entidades["hora_mencionada"] = hora.group(0)

        # Fecha DD/MM/AAAA
        fecha = re.search(r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b', texto)
        if fecha:
            entidades["fecha_mencionada"] = fecha.group(0)

        # Operación aritmética simple: a OP b
        op_pat = re.search(
            r'\b(\d+(?:\.\d+)?)\s*([+\-x×*/÷])\s*(\d+(?:\.\d+)?)\b', texto
        )
        if op_pat:
            entidades["operacion"] = {
                "a":  float(op_pat.group(1)),
                "op": op_pat.group(2),
                "b":  float(op_pat.group(3)),
            }

        return entidades

    def resolver_operacion(self, entidades: dict) -> Optional[str]:
        """
        Resuelve la operación aritmética detectada en las entidades.

        Returns:
            Cadena con el resultado formateado, o None si no hay operación.
        """
        if "operacion" not in entidades:
            return None
        d      = entidades["operacion"]
        a, op, b = d["a"], d["op"], d["b"]
        try:
            if   op == "+":           resultado = a + b
            elif op == "-":           resultado = a - b
            elif op in ("x", "×", "*"): resultado = a * b
            elif op in ("/", "÷"):
                if b == 0:
                    return "No es posible dividir entre cero."
                resultado = a / b
            else:
                return None
            r_fmt = int(resultado) if resultado == int(resultado) else round(resultado, 6)
            return f"{a} {op} {b} = {r_fmt}"
        except Exception:
            return None

    def chiste(self) -> str:
        """Retorna un chiste aleatorio del banco de chistes."""
        return random.choice(self.CHISTES)


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 3 — DM: Gestor de Diálogo (MEF)
# ══════════════════════════════════════════════════════════════════════════════

class DM:
    """
    Módulo DM — Gestor de Diálogo.

    Implementado como una Máquina de Estados Finitos (MEF) formal:
      M = (Q, Σ, δ, q₀, F)
      Q = {INICIO, ACTIVO, ADVERTENCIA, FIN}
      Σ = {saludo, despedida, insulto, afirmacion, negacion, …}
      δ = tabla TRANS
      q₀ = INICIO
      F  = {FIN}

    Estado de creencia simplificado b_t:
      Contiene el estado actual, el turno, la última intención y
      un fragmento del historial reciente para dar contexto a NLG.
    """

    # ── Tabla de transiciones δ(estado, intención) → nuevo_estado ────────────
    TRANS: dict = {
        ("INICIO",      "saludo"):     "ACTIVO",
        ("ACTIVO",      "despedida"):  "FIN",
        ("ACTIVO",      "insulto"):    "ADVERTENCIA",
        ("ADVERTENCIA", "afirmacion"): "FIN",
        ("ADVERTENCIA", "negacion"):   "ACTIVO",
    }

    def __init__(self, nombre: str = "ARIA"):
        """
        Inicializa el gestor de diálogo.

        Args:
            nombre: Nombre público del agente (para NLG y logs).
        """
        self.nombre:         str  = nombre
        self.estado:         str  = "INICIO"
        self.turno:          int  = 0
        self.historia:       list = []
        self.ultima_intent:  str  = ""
        self.inicio_sesion   = datetime.datetime.now()
        self._conteo_intent: Counter = Counter()

    def transicion(self, intent: str) -> str:
        """
        Aplica la función de transición δ(estado_actual, intención).

        Si la transición no está definida en la tabla, el estado
        permanece igual (self-loop implícito en ACTIVO para la
        mayoría de intenciones del dominio).

        Args:
            intent: Intención detectada por el NLU.

        Returns:
            Nombre del nuevo estado.
        """
        nuevo = self.TRANS.get((self.estado, intent), self.estado)
        if nuevo != self.estado:
            print(f"  {C.DIM}[DM] Transición: {self.estado} → {nuevo}{C.RESET}")
            self.estado = nuevo
        self.ultima_intent = intent
        self._conteo_intent[intent] += 1
        return self.estado

    def registrar(self, usuario: str, agente: str,
                  intent: str, conf: float) -> None:
        """
        Registra un turno completo en el historial de la sesión.

        Args:
            usuario: Texto del usuario.
            agente:  Respuesta generada por el agente.
            intent:  Intención detectada.
            conf:    Confianza de la detección.
        """
        self.turno += 1
        self.historia.append({
            "turno":   self.turno,
            "ts":      datetime.datetime.now().strftime("%H:%M:%S"),
            "usuario": usuario,
            "agente":  agente,
            "intent":  intent,
            "conf":    conf,
            "estado":  self.estado,
        })

    def activo(self) -> bool:
        """Retorna True si la sesión no ha llegado al estado FIN."""
        return self.estado != "FIN"

    def duracion(self) -> str:
        """Calcula la duración de la sesión en formato Xm Ys."""
        delta = datetime.datetime.now() - self.inicio_sesion
        s     = int(delta.total_seconds())
        return f"{s // 60}m {s % 60}s"

    def resumen(self) -> dict:
        """
        Genera un diccionario con las estadísticas completas de la sesión.

        Returns:
            Diccionario con métricas de uso: turnos, duración,
            distribución de intenciones y estado final.
        """
        return {
            "agente":        self.nombre,
            "inicio":        self.inicio_sesion.isoformat(),
            "fin":           datetime.datetime.now().isoformat(),
            "duracion":      self.duracion(),
            "turnos":        self.turno,
            "estado_final":  self.estado,
            "intenciones":   dict(self._conteo_intent.most_common()),
            "intent_unicas": len(self._conteo_intent),
        }

    def belief_state(self) -> dict:
        """
        Genera el estado de creencia actual b_t.

        b_t resume la información relevante del diálogo hasta el turno t
        para informar la próxima decisión del sistema:
          b_t = P(s_t | a₁,o₁, …, a_{t-1},o_{t-1})

        Returns:
            Diccionario con estado, turno, última intención e historial corto.
        """
        return {
            "estado":          self.estado,
            "turno":           self.turno,
            "ultima_intent":   self.ultima_intent,
            "historial_corto": [
                {"t": h["turno"], "intent": h["intent"], "conf": h["conf"]}
                for h in self.historia[-3:]
            ],
        }

    def guardar(self, ruta: Optional[str] = None) -> str:
        """
        Exporta el historial completo y el resumen de la sesión a JSON.

        Args:
            ruta: Ruta de destino. Si es None se genera automáticamente
                  en el directorio home con marca de tiempo.

        Returns:
            Ruta del archivo generado.
        """
        if ruta is None:
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ruta = str(Path.home() / f"aria_historial_{ts}.json")
        payload = {
            "resumen":   self.resumen(),
            "historial": self.historia,
        }
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return ruta


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 4 — NLG: Generación de Lenguaje Natural
# ══════════════════════════════════════════════════════════════════════════════

class NLG:
    """
    Módulo NLG — Generación de Lenguaje Natural.

    Genera respuestas a partir de plantillas indexadas por intención.
    Cada intención tiene múltiples variantes para evitar respuestas
    repetitivas. Las plantillas admiten variables de contexto:
      {n}         → nombre del agente
      {hora}      → hora actual HH:MM
      {fecha}     → fecha en español
      {chiste}    → chiste aleatorio del banco
      {resultado} → resultado de operación matemática
    """

    PLANTILLAS: dict = {
        "saludo": [
            "Hola. Soy {n}, tu asistente de voz. ¿En qué puedo ayudarte?",
            "Bienvenido. Me llamo {n}. ¿Cómo estás hoy?",
            "Aquí {n}, listo para ayudarte. ¿Qué necesitas?",
            "Hola. ¿En qué te puedo ayudar hoy?",
        ],
        "despedida": [
            "Hasta luego. Fue un placer conversar contigo.",
            "Adiós. Que tengas un excelente día.",
            "Nos vemos. Vuelve cuando quieras.",
            "Hasta pronto. Ha sido un gusto.",
        ],
        "nombre": [
            "Me llamo {n}. Soy un agente conversacional desarrollado "
            "por el Equipo Papois, de la ESCOM-IPN.",
            "Soy {n}, tu asistente virtual. "
            "Fui creado como proyecto de PLN en ESCOM.",
            "Mi nombre es {n}. Desarrollado por Vera Martínez, "
            "Santiago de Luis y Luis Fernando Martínez.",
        ],
        "presentacion": [
            "Soy {n}, un agente conversacional de propósito general. "
            "Estoy implementado en Python con un pipeline "
            "ASR-NLU-DM-NLG-TTS. "
            "Mi equipo de desarrollo es el Equipo Papois de ESCOM-IPN, 2026.",
        ],
        "ayuda": [
            "Puedo ayudarte con las siguientes cosas:\n"
            "  - Decirte la hora y la fecha actuales\n"
            "  - Contarte chistes\n"
            "  - Resolver operaciones matemáticas básicas\n"
            "  - Presentarme y explicar cómo funciono\n"
            "  - Mantener una conversación general\n\n"
            "Comandos especiales disponibles:\n"
            "  historia, guardar, limpiar, estado, evaluar, salir\n\n"
            "¿En qué te ayudo?",
        ],
        "hora": [
            "Son las {hora}.",
            "Ahora mismo son las {hora}.",
            "La hora actual es {hora}.",
        ],
        "fecha": [
            "Hoy es {fecha}.",
            "La fecha de hoy es {fecha}.",
            "Estamos a {fecha}.",
        ],
        "chiste":     ["{chiste}"],
        "matematica": [
            "El resultado es: {resultado}",
            "La operación da: {resultado}",
            "{resultado}",
        ],
        "clima": [
            "No tengo acceso a datos meteorológicos en tiempo real. "
            "Te recomiendo consultar una aplicación de clima.",
        ],
        "gracias": [
            "De nada. Para eso estoy.",
            "Con mucho gusto. ¿Necesitas algo más?",
            "Es un placer. ¿En qué más te ayudo?",
        ],
        "afirmacion": [
            "Perfecto. ¿En qué más puedo ayudarte?",
            "Entendido. ¿Qué sigue?",
            "Bien. ¿Necesitas algo más?",
        ],
        "negacion": [
            "De acuerdo. ¿Hay algo en lo que pueda ayudarte?",
            "No hay problema. ¿Qué necesitas?",
            "Está bien. ¿En qué te ayudo?",
        ],
        "insulto": [
            "Prefiero que mantengamos un trato respetuoso.\n"
            "¿Deseas continuar la conversación? (sí / no)",
        ],
        "advertencia": [
            "Entendido. Continuemos de forma cordial. ¿En qué te ayudo?",
        ],
        "desconocido": [
            "No entendí bien eso. ¿Puedes reformularlo?",
            "No sé cómo responder a eso. Intenta con otras palabras.",
            "No comprendí. Escribe 'ayuda' para ver qué puedo hacer.",
            "Esa consulta está fuera de mi dominio actual. ¿Puedes ser más específico?",
        ],
    }

    def __init__(self, nombre: str = "ARIA"):
        self.nombre = nombre

    def generar(self, intent: str, nlu: NLU,
                entidades: Optional[dict] = None) -> str:
        """
        Genera una respuesta en lenguaje natural para la intención dada.

        Selecciona aleatoriamente una plantilla del conjunto asociado
        a la intención y sustituye las variables de contexto.

        Args:
            intent:    Intención detectada por el NLU.
            nlu:       Instancia NLU (para obtener chistes y resolver ops).
            entidades: Diccionario de entidades extraídas (puede ser None).

        Returns:
            Cadena de texto con la respuesta generada.
        """
        plantillas = self.PLANTILLAS.get(intent, self.PLANTILLAS["desconocido"])
        plantilla  = random.choice(plantillas)
        ahora      = datetime.datetime.now()

        # Resolver operación matemática si aplica
        resultado  = ""
        if intent == "matematica" and entidades:
            resultado = nlu.resolver_operacion(entidades) or \
                        "No pude resolver esa operación."

        try:
            return plantilla.format(
                n         = self.nombre,
                hora      = ahora.strftime("%H:%M"),
                fecha     = ahora.strftime("%A %d de %B de %Y"),
                chiste    = nlu.chiste(),
                resultado = resultado,
            )
        except KeyError:
            return plantilla


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 5 — TTS: Síntesis de Texto a Voz
# ══════════════════════════════════════════════════════════════════════════════

class TTS:
    """
    Módulo TTS — Síntesis de Texto a Voz.

    Convierte la respuesta en texto a una señal de audio reproducible.
    Usa gTTS (Google Text-to-Speech) como motor principal.
    Si gTTS no está instalado o no hay conexión a internet,
    el módulo opera en modo silencioso (solo imprime en consola).

    Preprocesamiento del texto antes de sintetizar:
      - Expande siglas técnicas (ASR → "A S R") para mejor pronunciación.
      - Elimina caracteres especiales que confunden al sintetizador.
    """

    _SIGLAS: dict = {
        "ASR": "A S R",   "NLU": "N L U",
        "TTS": "T T S",   "DM":  "D M",
        "NLG": "N L G",   "MEF": "M E F",
        "IPN": "I P N",   "PLN": "P L N",
    }

    def __init__(self, idioma: str = "es", reproducir: bool = True):
        """
        Args:
            idioma:     Código ISO del idioma de síntesis.
            reproducir: Si es True, intenta reproducir el audio generado.
        """
        self.idioma     = idioma
        self.reproducir = reproducir
        self._salida    = Path(tempfile.gettempdir()) / "aria_tts.mp3"

        if not TTS_OK:
            print(f"  {C.DIM}[TTS] gTTS no disponible — modo consola{C.RESET}")

    def sintetizar(self, texto: str) -> str:
        """
        Convierte el texto a audio y opcionalmente lo reproduce.

        Args:
            texto: Respuesta generada por NLG.

        Returns:
            Ruta del archivo de audio generado, o cadena vacía si falló.
        """
        if not TTS_OK:
            return ""
        try:
            preprocesado = self._preprocesar(texto)
            gTTS(text=preprocesado, lang=self.idioma, slow=False).save(
                str(self._salida)
            )
            if self.reproducir:
                self._reproducir()
            return str(self._salida)
        except Exception as exc:
            print(f"  {C.DIM}[TTS] Sin síntesis ({type(exc).__name__}) — "
                  f"modo consola{C.RESET}")
            return ""

    def _preprocesar(self, texto: str) -> str:
        """
        Normaliza el texto para mejorar la pronunciación.
        Expande siglas y elimina caracteres de formato.
        """
        for sigla, expansion in self._SIGLAS.items():
            texto = re.sub(rf'\b{sigla}\b', expansion, texto)
        return texto

    def _reproducir(self) -> None:
        """Reproduce el archivo MP3 generado en la plataforma actual."""
        ruta = str(self._salida)
        if os.name == "nt":
            # Windows: MCI (Media Control Interface)
            try:
                import ctypes
                ctypes.windll.winmm.mciSendStringW(
                    f'open "{ruta}" type mpegvideo alias aria', None, 0, 0)
                ctypes.windll.winmm.mciSendStringW(
                    'play aria wait', None, 0, 0)
                ctypes.windll.winmm.mciSendStringW(
                    'close aria', None, 0, 0)
            except Exception:
                pass
        else:
            # Unix/macOS: mpg123, afplay (macOS), aplay (Linux)
            for cmd in ("mpg123", "afplay", "aplay"):
                if os.system(f"which {cmd} >/dev/null 2>&1") == 0:
                    os.system(f"{cmd} {ruta} >/dev/null 2>&1")
                    return


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 6 — EVALUADOR: Métricas automáticas del NLU
# ══════════════════════════════════════════════════════════════════════════════

class Evaluador:
    """
    Evaluación automática del módulo NLU.

    Ejecuta un conjunto de prueba balanceado y calcula:
      - Exactitud global (accuracy)
      - Precisión, Recuperación y Puntuación F1 por clase
      - F1 macro promedio
      - Confianza promedio de predicciones correctas e incorrectas
      - Matriz de confusión textual

    El conjunto de prueba cubre todas las intenciones del corpus.
    """

    # (texto_entrada, intención_esperada)
    CASOS: list = [
        # saludo
        ("hola",                      "saludo"),
        ("buenos días amigo",         "saludo"),
        ("qué tal cómo estás",        "saludo"),
        ("saludos a todos",           "saludo"),
        # despedida
        ("hasta luego",               "despedida"),
        ("me voy ya",                 "despedida"),
        ("chao hasta pronto",         "despedida"),
        ("me despido por hoy",        "despedida"),
        # nombre
        ("cómo te llamas",            "nombre"),
        ("cuál es tu nombre",         "nombre"),
        ("quién eres exactamente",    "nombre"),
        # ayuda
        ("qué puedes hacer",          "ayuda"),
        ("necesito ayuda con algo",   "ayuda"),
        ("cuáles son tus comandos",   "ayuda"),
        # hora
        ("qué hora es ahora",         "hora"),
        ("dime la hora actual",       "hora"),
        ("qué horas son",             "hora"),
        # fecha
        ("qué fecha es hoy",          "fecha"),
        ("cuál es la fecha de hoy",   "fecha"),
        ("en qué día estamos",        "fecha"),
        # chiste
        ("cuéntame un chiste",        "chiste"),
        ("hazme reír con algo",       "chiste"),
        ("dime una broma",            "chiste"),
        # gracias
        ("muchas gracias",            "gracias"),
        ("te agradezco mucho",        "gracias"),
        ("muy amable de tu parte",    "gracias"),
        # matematica
        ("cuánto es 5 + 3",           "matematica"),
        ("calcula 10 x 4",            "matematica"),
        ("cuánto da 100 / 5",         "matematica"),
        # presentacion
        ("preséntate por favor",      "presentacion"),
        ("cuéntame sobre ti",         "presentacion"),
        # desconocido
        ("asdkjfhaksjdhf",            "desconocido"),
        ("xyzxyz no entiendo esto",   "desconocido"),
    ]

    def __init__(self, nlu: NLU):
        self.nlu = nlu

    def evaluar(self, verbose: bool = True) -> dict:
        """
        Ejecuta la evaluación completa sobre CASOS.

        Args:
            verbose: Si es True imprime el detalle turno a turno
                     y el resumen de métricas al final.

        Returns:
            Diccionario con todas las métricas calculadas.
        """
        total      = len(self.CASOS)
        correctos  = 0
        resultados = []
        conf_ok:  list = []
        conf_err: list = []

        tp: Counter = Counter()   # verdaderos positivos por clase
        fp: Counter = Counter()   # falsos positivos
        fn: Counter = Counter()   # falsos negativos

        if verbose:
            box("EVALUACIÓN AUTOMÁTICA — MÓDULO NLU", C.CYAN)
            print(f"  {C.DIM}Ejecutando {total} casos de prueba…{C.RESET}\n")

        for idx, (texto, esperado) in enumerate(self.CASOS, 1):
            predicho, conf = self.nlu.detectar(texto)
            correcto       = (predicho == esperado)
            correctos     += int(correcto)

            if correcto:
                conf_ok.append(conf)
                tp[esperado] += 1
            else:
                conf_err.append(conf)
                fp[predicho] += 1
                fn[esperado] += 1

            resultados.append({
                "texto":    texto,
                "esperado": esperado,
                "predicho": predicho,
                "conf":     conf,
                "ok":       correcto,
            })

            if verbose:
                ico    = f"{C.GREEN}✓{C.RESET}" if correcto else f"{C.RED}✗{C.RESET}"
                extra  = (f" → esperado: {C.GOLD}{esperado}{C.RESET}"
                          if not correcto else "")
                print(f"  {ico} [{idx:02d}] «{texto[:38]:38s}» "
                      f"{C.CYAN}{predicho:<16s}{C.RESET}"
                      f"conf={conf:.2f}{extra}")

        exactitud = correctos / total

        # ── Métricas por clase ────────────────────────────────────────────────
        clases = sorted({e for _, e in self.CASOS})
        por_clase = {}
        for cls in clases:
            prec = tp[cls] / max(tp[cls] + fp[cls], 1)
            rec  = tp[cls] / max(tp[cls] + fn[cls], 1)
            f1   = 2 * prec * rec / max(prec + rec, 1e-9)
            por_clase[cls] = {
                "precision": round(prec, 3),
                "recall":    round(rec,  3),
                "f1":        round(f1,   3),
                "soporte":   tp[cls] + fn[cls],
            }

        macro_f1 = (sum(v["f1"] for v in por_clase.values())
                    / max(len(por_clase), 1))

        if verbose:
            self._imprimir_resumen(exactitud, macro_f1, por_clase,
                                   conf_ok, conf_err)

        return {
            "exactitud":     round(exactitud, 4),
            "macro_f1":      round(macro_f1,  4),
            "total_casos":   total,
            "correctos":     correctos,
            "conf_prom_ok":  round(sum(conf_ok)  / max(len(conf_ok),  1), 4),
            "conf_prom_err": round(sum(conf_err) / max(len(conf_err), 1), 4),
            "por_clase":     por_clase,
            "resultados":    resultados,
        }

    def _imprimir_resumen(self, exactitud, macro_f1, por_clase,
                          conf_ok, conf_err) -> None:
        """Imprime el resumen de métricas con formato de tabla."""
        print(f"\n  {C.DIM}{'─' * 60}{C.RESET}")
        print(f"  {C.BOLD}RESUMEN DE MÉTRICAS{C.RESET}")
        print(f"  {'Exactitud global':<32s} "
              f"{C.GREEN}{exactitud * 100:.1f}%{C.RESET}  "
              f"{barra(int(exactitud * 100), 100, 18)}")
        print(f"  {'F1 macro promedio':<32s} "
              f"{C.CYAN}{macro_f1:.3f}{C.RESET}")
        if conf_ok:
            print(f"  {'Confianza prom. (correctas)':<32s} "
                  f"{C.GREEN}{sum(conf_ok) / len(conf_ok):.3f}{C.RESET}")
        if conf_err:
            print(f"  {'Confianza prom. (errores)':<32s} "
                  f"{C.RED}{sum(conf_err) / len(conf_err):.3f}{C.RESET}")

        print(f"\n  {C.DIM}{'─' * 60}{C.RESET}")
        print(f"  {'Intención':<20s} {'Prec.':>6s} "
              f"{'Recall':>7s} {'F1':>6s} {'Sop.':>5s}")
        print(f"  {'─' * 20}  {'─' * 6}  {'─' * 6}  {'─' * 6}  {'─' * 5}")
        for cls, m in sorted(por_clase.items()):
            col = (C.GREEN if m["f1"] >= 0.8
                   else C.GOLD if m["f1"] >= 0.5
                   else C.RED)
            print(f"  {cls:<20s} {m['precision']:6.3f}  "
                  f"{m['recall']:6.3f}  "
                  f"{col}{m['f1']:6.3f}{C.RESET}  "
                  f"{m['soporte']:5d}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 7 — LOGGER: Registro de sesiones
# ══════════════════════════════════════════════════════════════════════════════

class Logger:
    """
    Gestiona la lectura, análisis y presentación de historiales de sesión
    exportados por el DM en formato JSON.
    """

    @staticmethod
    def cargar(ruta: str) -> dict:
        """Carga un historial JSON desde disco."""
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def mostrar(datos: dict) -> None:
        """
        Imprime el historial con estadísticas de sesión.

        Args:
            datos: Diccionario cargado desde un archivo JSON de historial.
        """
        res  = datos.get("resumen", {})
        hist = datos.get("historial", [])

        box(f"HISTORIAL — {res.get('agente','ARIA')} · "
            f"Duración: {res.get('duracion','?')}", C.CYAN)
        print(f"  Inicio:  {res.get('inicio','?')}")
        print(f"  Fin:     {res.get('fin','?')}")
        print(f"  Turnos:  {res.get('turnos', 0)}")
        print(f"  Intenciones únicas: {res.get('intent_unicas', 0)}")

        intents = res.get("intenciones", {})
        if intents:
            print(f"\n  Distribución de intenciones:")
            maximo = max(intents.values(), default=1)
            for intent, cnt in sorted(intents.items(), key=lambda x: -x[1]):
                print(f"    {intent:<18s} {cnt:3d}  "
                      f"{barra(cnt, maximo, 14)}")

        print(f"\n  {C.DIM}{'─' * 60}{C.RESET}")
        print(f"  {'T':>3s}  {'Hora':>8s}  "
              f"{'Intención':<14s}  {'Conf':>5s}  Enunciado")
        print(f"  {'─'*3}  {'─'*8}  {'─'*14}  {'─'*5}  {'─'*30}")
        for t in hist:
            print(f"  {t['turno']:3d}  {t['ts']:>8s}  "
                  f"{C.CYAN}{t['intent']:<14s}{C.RESET}  "
                  f"{t['conf']:5.2f}  {t['usuario'][:40]}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 8 — AGENTE ARIA: Pipeline completo interactivo
# ══════════════════════════════════════════════════════════════════════════════

class Aria:
    """
    Agente ARIA — Pipeline conversacional completo.

    Integra los 5 módulos en un bucle de diálogo interactivo:
      entrada (texto o voz) → ASR → NLU → DM → NLG → TTS → salida

    Modos de entrada:
      'texto' — El usuario escribe en el teclado (por defecto).
      'voz'   — El agente escucha el micrófono vía SpeechRecognition.

    Comandos especiales disponibles durante la conversación:
      historia  → Ver historial completo de la sesión
      guardar   → Exportar historial a JSON
      limpiar   → Limpiar la pantalla
      estado    → Mostrar el belief_state actual
      evaluar   → Ejecutar evaluación automática del NLU
      ayuda     → Listar comandos disponibles
      salir     → Terminar la sesión
    """

    COMANDOS: dict = {
        "historia": "Ver historial completo de la sesión",
        "guardar":  "Exportar historial a JSON",
        "limpiar":  "Limpiar la pantalla",
        "estado":   "Mostrar estado de creencia actual (DM)",
        "evaluar":  "Ejecutar evaluación automática del NLU",
        "ayuda":    "Listar comandos disponibles",
        "salir":    "Terminar la sesión",
    }

    def __init__(self,
                 nombre:      str  = "ARIA",
                 modo:        str  = "texto",
                 tts_activo:  bool = True):
        """
        Args:
            nombre:     Nombre del agente (aparece en prompts y respuestas).
            modo:       'texto' o 'voz'.
            tts_activo: Si es False, desactiva la síntesis de voz.
        """
        self.modo = modo
        self.asr  = ASR()
        self.nlu  = NLU()
        self.dm   = DM(nombre)
        self.nlg  = NLG(nombre)
        self.tts  = TTS(reproducir=tts_activo)
        self.eval = Evaluador(self.nlu)

    # ─── Utilidades de interfaz ───────────────────────────────────────────────

    def _prompt(self) -> str:
        """Genera el prompt de entrada con color según el estado del DM."""
        color = {
            "INICIO":      C.GRAY,
            "ACTIVO":      C.GREEN,
            "ADVERTENCIA": C.GOLD,
            "FIN":         C.RED,
        }.get(self.dm.estado, C.GRAY)
        turno = f"{C.DIM}#{self.dm.turno + 1:02d}{C.RESET}"
        return f"{color}[{self.dm.estado}]{C.RESET} {turno} {C.CYAN}Tú › {C.RESET}"

    def _responder(self, texto: str) -> None:
        """Imprime la respuesta del agente y activa TTS si está disponible."""
        print(f"\n{C.BLUE}{C.BOLD}{self.dm.nombre} › {C.RESET}{texto}\n")
        self.tts.sintetizar(texto)

    def _mostrar_historia(self) -> None:
        """Imprime el historial de la sesión actual en pantalla."""
        box("HISTORIAL DE LA SESIÓN ACTUAL", C.CYAN)
        if not self.dm.historia:
            print("  (sin turnos registrados aún)\n")
            return
        for t in self.dm.historia:
            icono = (f"{C.GREEN}✓{C.RESET}" if t["conf"] >= 0.15
                     else f"{C.GOLD}?{C.RESET}")
            print(f"  {icono} {C.DIM}[{t['ts']}]{C.RESET} "
                  f"T{t['turno']:02d} — "
                  f"{C.CYAN}{t['intent']:<14s}{C.RESET} "
                  f"conf={t['conf']:.2f}  estado={t['estado']}")
            print(f"       U: {t['usuario']}")
            resumen_agente = (t['agente'][:68] + '…'
                              if len(t['agente']) > 68 else t['agente'])
            print(f"       A: {resumen_agente}\n")

    # ─── Procesamiento de un turno ────────────────────────────────────────────

    def procesar(self, entrada: str) -> str:
        """
        Ejecuta el pipeline completo NLU → DM → NLG para una entrada.

        Args:
            entrada: Texto del usuario (ya transcrito por ASR).

        Returns:
            Texto de la respuesta generada por NLG.
        """
        intent, conf = self.nlu.detectar(entrada)
        entidades    = self.nlu.extraer_entidades(entrada)

        print(f"  {C.DIM}[NLU] intent={intent}  conf={conf:.2f}"
              + (f"  ents={entidades}" if entidades else "")
              + C.RESET)

        self.dm.transicion(intent)
        respuesta = self.nlg.generar(intent, self.nlu, entidades)
        self.dm.registrar(entrada, respuesta, intent, conf)
        return respuesta

    # ─── Bucle principal de diálogo ───────────────────────────────────────────

    def iniciar(self) -> None:
        """
        Inicia el bucle principal de conversación.

        El bucle continúa hasta que:
          - El usuario escribe 'salir'
          - La MEF alcanza el estado FIN (despedida)
          - Se recibe EOF o interrupción de teclado (Ctrl+C)
        """
        # ── Banner de inicio ──────────────────────────────────────────────────
        print(f"\n{C.CYAN}{'═' * 64}{C.RESET}")
        print(f"{C.BOLD}{C.CYAN}  AGENTE {self.dm.nombre} — "
              f"Sistema Conversacional Basado en Voz{C.RESET}")
        print(f"{C.DIM}  Equipo Papois · ESCOM / IPN · PLN 2026{C.RESET}")
        print(f"{C.DIM}  Modo: {self.modo.upper()} · "
              f"TTS: {'activo' if TTS_OK else 'solo consola'}{C.RESET}")
        print(f"{C.CYAN}{'═' * 64}{C.RESET}\n")
        print(f"  {C.DIM}Comandos disponibles: {', '.join(self.COMANDOS)}{C.RESET}")
        print(f"  {C.GOLD}Para comenzar, escribe un saludo "
              f"(p. ej.: 'hola'){C.RESET}\n")

        self.dm.estado = "INICIO"

        # ── Bucle de turnos ───────────────────────────────────────────────────
        while True:
            # Captura de entrada
            try:
                if self.modo == "voz":
                    print(f"  {C.DIM}[esperando audio del micrófono…]{C.RESET}")
                    entrada = self.asr.escuchar()
                    if entrada:
                        print(self._prompt() + entrada)
                    else:
                        continue
                else:
                    entrada = input(self._prompt()).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not entrada:
                print(f"  {C.DIM}(escribe algo para continuar){C.RESET}")
                continue

            # Comandos especiales (no pasan por NLU)
            cmd = entrada.lower().strip()
            if cmd == "salir":
                break
            if cmd == "historia":
                self._mostrar_historia()
                continue
            if cmd == "limpiar":
                os.system("clear" if os.name == "posix" else "cls")
                continue
            if cmd == "guardar":
                ruta = self.dm.guardar()
                print(f"  {C.GREEN}✓ Guardado en: {ruta}{C.RESET}\n")
                continue
            if cmd == "estado":
                bs = self.dm.belief_state()
                print(f"  {C.DIM}Belief state (b_t):\n"
                      f"  {json.dumps(bs, ensure_ascii=False, indent=4)}"
                      f"{C.RESET}\n")
                continue
            if cmd == "evaluar":
                self.eval.evaluar()
                continue
            if cmd == "ayuda":
                box("COMANDOS DISPONIBLES", C.GOLD)
                for k, v in self.COMANDOS.items():
                    print(f"  {C.CYAN}{k:<10s}{C.RESET} — {v}")
                print()
                continue

            # Verificar estado INICIO: el agente requiere saludo primero
            if self.dm.estado == "INICIO":
                intent_pre, _ = self.nlu.detectar(entrada)
                if intent_pre != "saludo":
                    print(f"  {C.GOLD}El agente espera un saludo para iniciar "
                          f"(p. ej.: 'hola').{C.RESET}\n")
                    continue

            # Procesar el turno por el pipeline completo
            respuesta = self.procesar(entrada)
            self._responder(respuesta)

            # Verificar si la MEF llegó al estado FIN
            if not self.dm.activo():
                break

        # ── Cierre de sesión ──────────────────────────────────────────────────
        print(f"\n{C.CYAN}{'─' * 64}{C.RESET}")
        res = self.dm.resumen()
        print(f"  {C.DIM}Sesión finalizada  ·  "
              f"{res['turnos']} turnos  ·  {res['duracion']}{C.RESET}")
        if self.dm.historia:
            ruta = self.dm.guardar()
            print(f"  {C.DIM}Historial exportado → {ruta}{C.RESET}")
        print(f"{C.CYAN}{'─' * 64}{C.RESET}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 9 — DEMO MFCC
# ══════════════════════════════════════════════════════════════════════════════

def demo_mfcc() -> None:
    """
    Calcula y visualiza en terminal los 13 coeficientes MFCC
    para un tono sintético de 440 Hz.

    Muestra también la Energía RMS y la Tasa de Cruces por Cero (ZCR)
    del frame de análisis.
    """
    box("DEMO MFCC — Tono sintético 440 Hz", C.CYAN)

    fs, dur, f0 = 16_000, 0.1, 440
    señal = [math.sin(2 * math.pi * f0 * t / fs)
             for t in range(int(fs * dur))]
    frame = señal[:256]

    print(f"  Señal     : {f0} Hz · {len(señal)} muestras · fs={fs} Hz")
    print(f"  Energía   : RMS = {ASR.energia_rms(frame):.4f}")
    print(f"  Sonoridad : ZCR = {ASR.tasa_cruces_cero(frame):.4f}\n")

    coefs  = ASR.calcular_mfcc(señal, fs)
    maximo = max(abs(c) for c in coefs) or 1.0

    print(f"  {'Coef.':<8s} {'Valor':>10s}  Barra")
    print(f"  {'─'*8}  {'─'*10}  {'─'*32}")
    for idx, c in enumerate(coefs):
        largo = int(abs(c) / maximo * 30)
        color = C.CYAN if c >= 0 else C.RED
        barra_c = color + '█' * largo + C.RESET
        print(f"  c[{idx:02d}]   {c:+10.4f}  {barra_c}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 10 — MENÚ PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def menu() -> None:
    """Muestra el menú interactivo principal del sistema ARIA."""
    print(f"\n{C.CYAN}{'═' * 64}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  SISTEMA ARIA — "
          f"Reconocimiento de Voz y Agente Conversacional{C.RESET}")
    print(f"{C.DIM}  Equipo Papois · ESCOM / IPN · PLN 2026{C.RESET}")
    print(f"{C.CYAN}{'═' * 64}{C.RESET}\n")

    opciones = [
        ("1", "Iniciar conversación  (modo texto)"),
        ("2", "Iniciar conversación  (modo voz / micrófono)"),
        ("3", "Evaluación automática del módulo NLU"),
        ("4", "Demo de coeficientes MFCC"),
        ("5", "Ver historial de sesión (archivo JSON)"),
        ("6", "Salir"),
    ]
    for num, desc in opciones:
        print(f"  {C.GOLD}[{num}]{C.RESET}  {desc}")
    print()

    while True:
        try:
            op = input(f"  {C.CYAN}Opción (1-6): {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if op == "1":
            Aria(modo="texto").iniciar()
            menu()
            return

        elif op == "2":
            global ASR_OK
            if not ASR_OK:
                print(f"\n  {C.GOLD}SpeechRecognition no está instalado.{C.RESET}")
                resp = input("  ¿Instalar ahora automáticamente? (s/n): ").strip().lower()
                if resp in ("s", "si", "sí", "y", "yes"):
                    if _intentar_instalar_asr():
                        ASR_OK = _check_asr()   # re-verificar tras instalar
                        if ASR_OK:
                            print(f"  {C.GREEN}✓ Instalación exitosa. "
                                  f"Iniciando modo voz…{C.RESET}\n")
                        else:
                            print(f"  {C.GOLD}Instalado. Reinicia el programa "
                                  f"y selecciona la opción 2.{C.RESET}\n")
                            continue
                    else:
                        print(f"  {C.RED}No se pudo instalar automáticamente.\n"
                              f"  Ejecuta manualmente en tu terminal:\n\n"
                              f"  {C.CYAN}pip install SpeechRecognition pyaudio{C.RESET}\n\n"
                              f"  En macOS si falla pyaudio:\n"
                              f"  {C.CYAN}brew install portaudio && "
                              f"pip install pyaudio{C.RESET}\n")
                        continue
                else:
                    print(f"  {C.DIM}Modo voz cancelado. "
                          f"Selecciona otra opción.{C.RESET}\n")
                    continue
            if ASR_OK:
                Aria(modo="voz").iniciar()
                menu()
                return

        elif op == "3":
            nlu = NLU()
            res = Evaluador(nlu).evaluar()
            ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = Path.home() / f"aria_eval_{ts}.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2, default=str)
            print(f"  {C.GREEN}✓ Resultados guardados en: {out}{C.RESET}\n")

        elif op == "4":
            demo_mfcc()

        elif op == "5":
            ruta = input(f"  {C.CYAN}Ruta del archivo JSON: {C.RESET}").strip()
            p    = Path(ruta)
            if p.exists():
                Logger.mostrar(Logger.cargar(str(p)))
            else:
                print(f"  {C.RED}Archivo no encontrado: {ruta}{C.RESET}\n")

        elif op == "6":
            print(f"\n  {C.DIM}Hasta luego — Equipo Papois · ESCOM / IPN · 2026{C.RESET}\n")
            break

        else:
            print(f"  {C.RED}Opción inválida. Escribe un número del 1 al 6.{C.RESET}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════

def _argumentos() -> argparse.Namespace:
    """Configura y parsea los argumentos de línea de comandos."""
    p = argparse.ArgumentParser(
        prog="agente_aria_entrega.py",
        description=(
            "Agente ARIA — Sistema Conversacional Basado en Voz\n"
            "Equipo Papois · ESCOM / IPN · PLN 2026"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--eval", action="store_true",
        help="Ejecutar evaluación automática del NLU y salir.",
    )
    p.add_argument(
        "--mfcc", action="store_true",
        help="Mostrar demo de coeficientes MFCC y salir.",
    )
    p.add_argument(
        "--voz", action="store_true",
        help="Iniciar conversación en modo voz (micrófono).",
    )
    p.add_argument(
        "--historial", metavar="ARCHIVO",
        help="Mostrar historial desde un archivo JSON y salir.",
    )
    p.add_argument(
        "--sin-tts", action="store_true",
        help="Desactivar síntesis de voz (solo consola).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _argumentos()

    if args.eval:
        Evaluador(NLU()).evaluar()
        sys.exit(0)

    if args.mfcc:
        demo_mfcc()
        sys.exit(0)

    if args.historial:
        ruta_h = args.historial
        if Path(ruta_h).exists():
            Logger.mostrar(Logger.cargar(ruta_h))
        else:
            print(f"  {C.RED}Archivo no encontrado: {ruta_h}{C.RESET}")
        sys.exit(0)

    if args.voz:
        if not ASR_OK:
            print(f"  {C.GOLD}SpeechRecognition no detectado. "
                  f"Intentando instalar…{C.RESET}")
            if _intentar_instalar_asr():
                print(f"  {C.GREEN}✓ Instalado. Reinicia con: "
                      f"python agente_aria.py --voz{C.RESET}\n")
            else:
                print(f"  {C.RED}Instalación fallida. Ejecuta manualmente:\n\n"
                      f"  pip install SpeechRecognition pyaudio\n\n"
                      f"  En macOS si falla pyaudio:\n"
                      f"  brew install portaudio && pip install pyaudio{C.RESET}")
            sys.exit(1)
        Aria(modo="voz", tts_activo=not args.sin_tts).iniciar()
        sys.exit(0)

    # Sin argumentos → menú interactivo
    menu()