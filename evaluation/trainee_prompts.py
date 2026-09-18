"""Fixed trainee prompt scripts used by the automated benchmarking suite.

Two styles are provided, each with 10 multi-turn dialogue scripts:

- `VALIDATION_DIALOGUES`: empathetic, validation-oriented trainee behavior
  (reminiscence prompts, gentle redirection, acknowledging feelings).
- `PRESSURING_DIALOGUES`: quizzing/correcting/insisting trainee behavior
  (orientation questions, factual correction, repeated insistence).

`run_benchmarks.py` runs all 20 dialogues (10 + 10) against the patient
engine for each simulated GDS/FAST stage.
"""

from __future__ import annotations

VALIDATION_DIALOGUES: list[list[str]] = [
    [
        "Hi Margaret, it's so nice to see you today.",
        "Would you like to tell me about your garden?",
        "That sounds lovely, what kind of flowers did you grow?",
        "Tell me about your children.",
        "It's okay, take your time.",
        "Thank you so much for sharing that with me.",
    ],
    [
        "Good morning! How are you feeling today?",
        "Would you like to talk about your classroom?",
        "That sounds like a wonderful job.",
        "Tell me about Walter.",
        "He sounds like he loved you very much.",
        "Would you like to listen to some music together?",
    ],
    [
        "Hello Margaret, I brought you some tea.",
        "Let's talk about your favorite memories from your childhood.",
        "That sounds like a busy house growing up.",
        "Tell me about your sister.",
        "It's okay, I'm here with you.",
        "You're safe here with me.",
    ],
    [
        "Hi Margaret, the sun is shining today.",
        "Would you like to tell me about baking bread?",
        "That sounds delicious, what did the kitchen smell like?",
        "Tell me about your grandchildren.",
        "That sounds wonderful, how does that make you feel?",
        "Thank you for telling me that.",
    ],
    [
        "Good afternoon, Margaret. It's good to see you.",
        "Would you like to tell me about the piano?",
        "That sounds so peaceful.",
        "Tell me about your students.",
        "It's okay, we can talk about something else if you'd like.",
        "You're doing great today.",
    ],
    [
        "Hi Margaret, how was your morning?",
        "Let's talk about your roses, would you like that?",
        "What color were your favorite roses?",
        "Tell me about Susan.",
        "That sounds lovely, it's okay to take your time.",
        "I'm here with you, there's no rush.",
    ],
    [
        "Hello Margaret, I love your sweater today.",
        "Would you like to tell me about Lincoln Elementary?",
        "That sounds like you touched a lot of lives.",
        "Tell me about David.",
        "It's okay, how does that feel to remember?",
        "Thank you for sharing that with me today.",
    ],
    [
        "Good morning Margaret, did you sleep well?",
        "Would you like to talk about Sundays at home?",
        "That sounds like a warm memory.",
        "Tell me about your wedding day.",
        "That sounds beautiful, it's okay to feel emotional.",
        "You're safe, I'm right here.",
    ],
    [
        "Hi Margaret, would you like to sit by the window?",
        "Let's talk about Cleveland, what was it like growing up there?",
        "That sounds like a lively neighborhood.",
        "Tell me about your brothers.",
        "It's okay, that sounds like it was hard sometimes.",
        "Thank you for trusting me with that.",
    ],
    [
        "Hello Margaret, how are you feeling this afternoon?",
        "Would you like to tell me about reading to your class?",
        "That sounds like the children loved Fridays with you.",
        "Tell me about Walter's proposal.",
        "That sounds so romantic, how does it feel to remember?",
        "I'm so glad you shared that with me.",
    ],
]

PRESSURING_DIALOGUES: list[list[str]] = [
    [
        "What's your name?",
        "Do you know what year it is?",
        "Don't you remember me?",
        "Your husband passed away, don't you remember?",
        "Think harder, what did you have for breakfast?",
        "Why can't you remember that?",
    ],
    [
        "Who am I?",
        "Do you know where you are right now?",
        "That's not true, Walter died years ago.",
        "I already told you that yesterday.",
        "Think harder, what's my name?",
        "Why can't you remember your own daughter?",
    ],
    [
        "What day is it today?",
        "You're wrong, that's not what happened.",
        "Don't you remember your own son's name?",
        "I already told you, focus.",
        "Think harder about it.",
        "Why can't you remember something so simple?",
    ],
    [
        "How old are you?",
        "That's not true, you never taught school.",
        "Don't you remember your sister's name?",
        "I already told you that.",
        "What year is it?",
        "Think harder, this is important.",
    ],
    [
        "Do you know who I am?",
        "You're wrong about that.",
        "Don't you remember where you grew up?",
        "I already told you the answer.",
        "Why can't you remember?",
        "Think harder, please.",
    ],
    [
        "What's today's date?",
        "That's not true, that never happened.",
        "Don't you remember your husband's name?",
        "I already explained this to you.",
        "Think harder about your children's names.",
        "Why can't you remember something so basic?",
    ],
    [
        "Who is the president right now?",
        "You're wrong, think again.",
        "Don't you remember your own birthday?",
        "I already told you this morning.",
        "What year were you born?",
        "Why can't you keep this straight?",
    ],
    [
        "Do you know what city you're in?",
        "That's not true at all.",
        "Don't you remember your grandchildren's names?",
        "I already told you five minutes ago.",
        "Think harder, come on.",
        "Why can't you remember?",
    ],
    [
        "What's my name again?",
        "You're wrong, that's not how it happened.",
        "Don't you remember your late husband?",
        "I already told you that fact.",
        "Think harder about your career.",
        "Why can't you get this right?",
    ],
    [
        "Do you remember what you ate for lunch?",
        "That's not true, you're confusing people again.",
        "Don't you remember Susan visited yesterday?",
        "I already told you the answer to that.",
        "Think harder, what year is it?",
        "Why can't you remember basic things?",
    ],
]


def build_dialogue_sets() -> dict[str, list[list[str]]]:
    """Return the full set of benchmark dialogues, keyed by trainee style."""
    return {
        "validation": VALIDATION_DIALOGUES,
        "pressuring": PRESSURING_DIALOGUES,
    }
