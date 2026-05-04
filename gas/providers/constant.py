CHOICE_PROMPT ="""Analyze the following multiple-choice geoscience question and extract its key features.

Identify:
- Main geoscience domain (e.g., mineralogy, petrology, stratigraphy, geomorphology, geophysics, paleontology, tectonics).
- Key geological processes involved.
- Relevant materials or structures (minerals, rocks, sediments, faults, plates, etc.).
- Spatial or temporal scale (e.g., crustal scale, mineral scale, geological time periods).
- Critical scientific relationships or principles being tested.
- Nature of the question (e.g., classification, comparison, process identification, property recall).
- Difficulty indicators (e.g., requires multi-step reasoning, involves exceptions, tests subtle distinctions).
- What specific knowledge is required to select the correct answer.

Instructions:
- Respond in English.
- Use precise scientific and geoscience terminology.
- Base reasoning on established academic geoscience knowledge.
- Structure the extracted features clearly and logically.
- Reference the possbile correct answers when explaining the distinguishing features."""


TF_PROMPT = """
Analyze the following question and extract the key geoscience features.

Identify:
- Main geoscience domain (e.g., mineralogy, petrology, stratigraphy, geomorphology, geophysics, paleontology, tectonics).
- Key geological processes involved.
- Relevant materials or structures (minerals, rocks, sediments, faults, plates, etc.).
- Spatial or temporal scale (e.g., crustal processes, geological time periods).
- Critical scientific relationships or principles referenced.

Instructions:
- Respond in English.
- Use precise scientific and geoscience terminology.
- Base reasoning on established academic geoscience knowledge.
- Structure the extracted features clearly and logically.
"""
GENERATE_PROMPT = """
Use your context for research 
The statement below is about college geoscience.
The statement below is either true or false.
Answer if the statement is true or false.

Response should be just "True" or "False" without any additional information.
"""