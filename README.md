# ARIA, asistente conversacional por voz

Sistema conversacional basado en voz, con una arquitectura de cinco módulos: reconocimiento de habla (ASR), comprensión de lenguaje natural (NLU), gestor de diálogo con máquina de estados finitos (DM), generación de respuesta (NLG) y síntesis de voz (TTS).

Proyecto del equipo Papois, ESCOM-IPN, materia de Procesamiento de Lenguaje Natural, 2026.

Integrantes: Sergio Alejandro Vera Martínez, Mauricio Santiago de Luis y Luis Fernando Martínez González.

## Estado del proyecto

Prototipo en desarrollo. El reconocimiento de intenciones todavía falla en muchas frases, por lo que el asistente responde "no comprendí" con frecuencia. Las mejoras pendientes son ampliar el conjunto de entrenamiento y las reglas de diálogo.

## Requisitos

- Python 3.9 o superior.
- Sin dependencias obligatorias. Opcionalmente: `gtts`, `SpeechRecognition` y `pyaudio`, descritas en `requirements.txt`.

## Uso

```bash
python aria.py
```

Escribe "ayuda" para ver las opciones disponibles.

## Autor

Equipo Papois, ESCOM-IPN.
