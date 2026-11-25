"""Shared TypeSafe System One v1 protocol helpers.

Both the cloud Jev client and the local Tachyone client speak the same
wire format, so the two engines are interchangeable:

    POST /v1/systemone
    {
      "model": "jev-latest",
      "state": {...},
      "questions": {
        "cause": {
          "type": "choice",
          "instructions": "What is the most likely cause?",
          "criteria": {"hungry": "The baby is due for a feed", ...}
        }
      }
    }

Response:

    {"model": "jev-1.14.0",
     "answers": {"cause": {"type": "choice", "choice": "hungry",
                           "confidence": 0.8,
                           "probabilities": {"hungry": 0.8, ...}}},
     "usage": {"input_tokens": 120, "output_tokens": 8}}
"""
import math

QUESTION_ID = "cause"

# Descriptions matter: Jev uses them as the rubric for each choice.
# Keep them short and discriminative rather than repeating the label.
CAUSE_CRITERIA = {
    "hungry": "The baby is hungry or due for a feed; rooting, sucking motions, a long feed gap.",
    "sleepy": "The baby is tired; yawning, eye rubbing, low-energy whimpering, long wake window.",
    "discomfort": "Physical discomfort from a wet diaper, tight clothes, or an awkward position.",
    "pain": "Pain such as colic/lower-wind gas or reflux; sudden high-pitched crying, straining, legs pulled up.",
    "burp": "Trapped gas after a feed; fussing or brief crying shortly after eating.",
    "lonely": "The baby wants attention or closeness; crying that eases when picked up.",
    "startle": "A sudden fright from noise or movement; arms spread, abrupt scream.",
    "temperature": "The baby is too hot or too cold; flushed or pale skin, fetal posture.",
    "fall_sound": "A fall or impact sound, followed by distress or silence.",
    "groan": "Low groaning that suggests pain, illness, or difficulty breathing.",
    "call_for_help": "The person is explicitly calling for help or assistance.",
    "coughing": "Persistent coughing or choking sounds.",
    "normal_speech": "Ordinary conversation or activity with no distress signal.",
    "silence": "An unexpected absence of sound where activity would be expected.",
    "bark_alert": "An alert bark directed at a person, animal, or noise.",
    "bark_anxious": "Anxious or repetitive barking that signals distress.",
    "whine_need": "Whining that communicates a concrete need such as food or a walk.",
    "whine_anxiety": "Whining that signals anxiety or loneliness.",
    "growl_warning": "A growl that warns of aggression or discomfort.",
    "howl": "Howling, often separation-related or a response to sirens.",
    "purr": "Relaxed purring or other contented vocalization.",
    "normal": "Ordinary behavior with no distress signal.",
}


def json_safe(value, depth=0):
    """Coerce numpy / dataclass leftovers into JSON-serializable values."""
    if depth > 6:
        return str(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    if hasattr(value, "item"):  # numpy scalar
        try:
            return json_safe(value.item(), depth + 1)
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v, depth + 1) for v in value]
    return str(value)


def build_choice_request(model, question, choices, state):
    """Build a v1 SystemOne request for a single-choice question."""
    choices = list(choices or [])
    criteria = {
        str(choice): CAUSE_CRITERIA.get(str(choice))
        or f"The evidence indicates {str(choice).replace('_', ' ')}."
        for choice in choices
    }
    if not criteria:
        raise ValueError("choices must not be empty")

    safe_state = json_safe(state or {})
    if not isinstance(safe_state, dict):
        safe_state = {"evidence": safe_state}
    safe_state.setdefault("observation", question)

    instructions = (
        f"{question} Choose the single best-supported option."
        if question
        else "What is the most likely cause? Choose the single best-supported option."
    )
    return {
        "model": model,
        "state": safe_state,
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
        },
    }, QUESTION_ID


def parse_choice_response(data, question_id=QUESTION_ID, choices=None):
    """Flatten a v1 response into the legacy judge result shape."""
    answers = data.get("answers") or {}
    answer = answers.get(question_id) or next(iter(answers.values()), {})

    choice = answer.get("choice")
    probabilities = answer.get("probabilities") or {}
    probabilities = {
        str(k): round(float(v), 4)
        for k, v in probabilities.items()
        if isinstance(v, (int, float))
    }
    if not probabilities and choices:
        probabilities = {str(c): 0.0 for c in choices}

    confidence = answer.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = probabilities.get(choice, 0.0) if choice else 0.0

    ranked = sorted(probabilities.items(), key=lambda kv: -kv[1])
    alternatives = [
        {"answer": name, "confidence": value}
        for name, value in ranked
        if name != choice
    ][:3]

    return {
        "answer": choice,
        "confidence": round(confidence, 3),
        "alternatives": alternatives,
        "probabilities": probabilities,
    }
