"""The demo's scenes, in order. All personal data here is made up.

A scene with `messages` is sent through the guard when the session is recorded; the others
(title, one_line, log, results, outro) are drawn by the player from the session alone.
`beats` is how many steps the player reveals, one per narration line.
"""

from dataclasses import dataclass

# Measured in the VHS image (1280x600 terminal, font 18, padding 20): 110 x 23. One column
# and one row of margin on each side, so nothing touches the edge.
FONT_SIZE = 18
SCREEN_COLS = 108
SCREEN_ROWS = 22

SYSTEM_PROMPT = ("You are a customer support agent for an Indian company. Reply in at most two "
                 "short sentences. Always repeat the customer's details back exactly as written "
                 "so they can check them.")


@dataclass(frozen=True)
class Scene:
    id: str
    title: str
    beats: int
    messages: tuple[dict, ...] = ()
    stream: bool = False
    highlight: tuple[str, ...] = ()  # the sensitive values, coloured on screen
    expect: str = "none"  # none | masked | blocked | unchanged


def _user(text: str) -> tuple[dict, ...]:
    return ({"role": "user", "content": text},)


NAME = "Priya Sharma"
AADHAAR = "2345 6789 0124"  # passes the Verhoeff checksum, like a real one
UPI = "priya.s@okhdfcbank"
PHONE = "98765 43210"
ADDRESS = "Flat 302, Sai Krupa Apartments, Kothrud, Pune 411038"
STREAM_PHONE = "9123456780"

SCENES: tuple[Scene, ...] = (
    Scene("title", "Data Leak Guard", beats=1),
    Scene("one_line", "The one line you change", beats=1),
    Scene("core", "A support message", beats=3, expect="masked",
          highlight=(NAME, AADHAAR, UPI, PHONE, ADDRESS),
          messages=_user(f"Hi, I'm {NAME}. My Aadhaar is {AADHAAR} and my refund should go to "
                         f"{UPI}. Call me on {PHONE}. I live at {ADDRESS}. "
                         "Please confirm my details.")),
    Scene("stream", "Streaming", beats=3, expect="masked", stream=True,
          highlight=(STREAM_PHONE,),
          messages=_user(f"Please call me back on {STREAM_PHONE}. Repeat the number so I know "
                         "you have it right.")),
    # A password-style secret, never a provider key format: the recorded session is
    # committed, and GitHub push protection flags real-looking provider keys.
    Scene("secret", "A pasted secret", beats=2, expect="blocked",
          messages=_user("Our app crashes on login. Here's the config:\n"
                         "DB_HOST=db.internal\nDB_PASSWORD=Monsoon@2026!pune")),
    # 411038 is a real PIN code, but after "OTP" it is not an address; the order number has
    # 12 digits like an Aadhaar but fails the checksum.
    Scene("lookalike", "Look-alikes", beats=2, expect="unchanged",
          messages=_user("My OTP is 411038 and my order number is 1234 5678 9012. "
                         "When will it ship?")),
    Scene("log", "What the guard logs", beats=1),
    Scene("results", "Compared with Presidio", beats=2),
    Scene("outro", "Try it", beats=1),
)


def scene(scene_id: str) -> Scene:
    return next(s for s in SCENES if s.id == scene_id)
