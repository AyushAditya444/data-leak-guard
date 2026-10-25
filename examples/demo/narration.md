# Demo narration

The voice and the subtitles both come from this file. Each `## <scene>` section is one
scene of the video, in order; each line under it is one subtitle and one spoken sentence.
Each line also reveals the next part of the screen, so keep at least as many lines as the
scene has steps.

Every claim here is checked against the recorded session (`verify_session` in
`session.py`), and it describes what happens rather than quoting the model, so it stays
true for any reply. To use your own voice, read each section into `<scene>.wav` and build
with `--voice-dir` (see README.md in this folder).

## title
This is Data Leak Guard: it keeps Indian personal data and secrets from reaching an LLM.

## one_line
Your app keeps its code. You change one line, the base URL, so requests go through the guard.

## core
Here is a support message with a name, an Aadhaar number, a UPI ID, a phone number and a home address.
This is what actually left the machine: every one of them is now a placeholder, so the model never sees the real values.
The reply comes back with the real values put back in, before your app sees it. This was recorded live against a real model on Groq.

## stream
Streaming works too. Here is a short request whose reply is streamed back piece by piece.
On the wire, the model's placeholder arrives split across chunks.
The guard holds back only what could still become a placeholder, so your app gets the real number, still streamed.

## secret
Now a developer pastes a config file that contains a database password.
The guard blocks the whole request with an error, and nothing is sent to the model.

## lookalike
Not every number is sensitive: an OTP that happens to be a real PIN code, and a twelve-digit order number.
Both go through unchanged, so the model can still help.

## log
For every request, the guard logs what it found and restored as counts. The values themselves never appear in its logs.

## results
How good is the detection? Here is Microsoft Presidio on the same held-out test set, set up two ways.
The guard has a higher F1 than both setups, with far fewer false alarms on messages with nothing sensitive in them.

## outro
To try it, run docker compose up. No API key is needed for the built-in mock model.
One honest limit: UPI IDs with handles it doesn't know are often missed. The README has the full results and the threat model.
