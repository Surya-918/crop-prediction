import os

from groq import Groq


class TreatmentServiceError(RuntimeError):
    """Raised when treatment guidance cannot be provided."""


def get_disease_solution(disease_name: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise TreatmentServiceError(
            "Set GROQ_API_KEY in your environment to request treatment guidance."
        )

    client = Groq(api_key=api_key)
    prompt = f"""
Act as an expert agriculturalist and plant pathologist.
A plant has been diagnosed with the disease: '{disease_name.replace("_", " ")}'.
Please provide:
1. A brief description of the disease.
2. Primary causes.
3. Three practical step-by-step treatments or cures.
Keep the response concise, easy to understand, and format it with bullet points.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512,
    )
    solution = response.choices[0].message.content
    if not solution:
        raise TreatmentServiceError("Groq returned an empty treatment recommendation.")
    return solution
