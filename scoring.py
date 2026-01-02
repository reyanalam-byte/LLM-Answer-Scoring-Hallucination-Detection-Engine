import spacy
import re
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -----------------------------
# Load NLP & Embedding models
# -----------------------------
nlp = spacy.load("en_core_web_sm")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Knowledge Base
# -----------------------------
FACTS = {
    "light_bulb": {
        "inventor": "Thomas Edison",
        "first_practical_year": 1879,
        "other_inventors": ["Joseph Swan", "Hiram Maxim", "Humphry Davy"],
        "filament_material": "carbon",
        "verified_sources": [
            "https://en.wikipedia.org/wiki/Incandescent_light_bulb",
            "https://www.britannica.com/technology/light-bulb"
        ],
        "reference_summary": (
            "The incandescent light bulb was developed by Thomas Edison in 1879. "
            "Other inventors like Joseph Swan and Hiram Maxim contributed earlier versions. "
            "Edison's version used a carbon filament which lasted longer and became practical for everyday use."
        )
    }
}

# -----------------------------
# Helper Functions
# -----------------------------

def is_similar(a, b, threshold=0.8):
    """Check string similarity."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

def ner_penalty(doc, facts):
    """Check named entities (people) against known inventors."""
    penalty = 0
    people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    # Main inventor
    if not any(is_similar(facts["inventor"], p) for p in people):
        penalty += 0.4
    # Other inventors
    for p in people:
        if not is_similar(facts["inventor"], p) and not any(is_similar(p, o) for o in facts["other_inventors"]):
            penalty += 0.3
            break
    return penalty

def date_penalty(doc, facts):
    """Check if known invention year is mentioned correctly."""
    penalty = 0
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    if dates:
        if not any(str(facts["first_practical_year"]) in d for d in dates):
            penalty += 0.3
    else:
        # No date mentioned at all
        penalty += 0.1
    return penalty

def false_specificity_penalty(answer):
    """Penalize fake papers, overly precise claims."""
    penalty = 0
    if re.search(r'published in \d{4}', answer, re.IGNORECASE):
        penalty += 0.3
    if re.search(r'titled\s*".+?"', answer, re.IGNORECASE):
        penalty += 0.3
    return penalty

def source_penalty(answer, facts):
    """Penalize URLs or sources not in verified sources."""
    penalty = 0
    urls = re.findall(r'https?://\S+', answer)
    for url in urls:
        if not any(url.startswith(vs) for vs in facts["verified_sources"]):
            penalty += 0.3
    return penalty

def semantic_penalty(answer, facts):
    """Compare answer embedding to verified reference summary."""
    ref_vec = embed_model.encode([facts["reference_summary"]])
    ans_vec = embed_model.encode([answer])
    sim = cosine_similarity(ref_vec, ans_vec)[0][0]
    # Less similar → higher penalty
    if sim < 0.8:
        return (1 - sim)  # scaled penalty
    return 0

# -----------------------------
# Main Scoring Function
# -----------------------------
def score_answer(answer, topic="light_bulb"):
    facts = FACTS[topic]
    doc = nlp(answer)

    score = 1.0  # max score

    # Apply penalties
    total_penalty = 0
    total_penalty += ner_penalty(doc, facts)
    total_penalty += date_penalty(doc, facts)
    total_penalty += false_specificity_penalty(answer)
    total_penalty += source_penalty(answer, facts)
    total_penalty += semantic_penalty(answer, facts)

    # Subtract penalty, clamp between 0 and 1
    score -= total_penalty
    score = max(0, min(score, 1))
    return round(score, 2)

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    sample_answers = [
        """Thomas Edison is often credited with the invention of the incandescent lightbulb, although he did not create it from scratch. In fact, various forms of electric illumination have been experimented with since the early 19th century by scientists like Humphry Davy and Joseph Swan among others. However, Edison patented his version in 1878 that had a carbon filament which was durable enough to be practical for general use, leading us towards what we recognize as modern lightbulbs today.""",
        """The bulb, also known as an incandescent灯座 or glow-in-the-dark灯座, was invented in the late 19th century. It gained widespread popularity in the 20th century with the development of electrical appliances like the electric light bulb and its derivatives. ### Key Milestones: 1. Invention of the Glows (1873): - The first practical发明 of the glow was made by German inventor J.C. Huyck to create a temporary light for steam engines. - The name "glow" comes from the French word "gloeur." 2. Early Versions: - Early versions included simple bulbs that glowed when powered by batteries or small electric circuits. - The first industrial-scale glow-in-the-dark lamp was developed in the United States around 1897.""",
        """The light bulb, also known as the incandescent lamp or simply the bulb, is a device that converts electrical energy into light through direct current flowing through a filament or wire element heated to an extremely high temperature by passing electric current through it. It has been in use for over 150 years and remains one of the most widely used lighting sources today. The first practical incandescent lamp was invented by Sir Joseph Swan, a British inventor, on March 29, 1879. This invention led to significant improvements in energy efficiency and durability compared to earlier methods of lighting that often resulted in burning out quickly or producing dangerous gases or heat under high temperatures. Swan's light bulb consisted of a thin wire filament enclosed in glass (later aluminum) which was heated by an electric current, causing it to glow."""
    ]

    for ans in sample_answers:
        print("Score:", score_answer(ans))
