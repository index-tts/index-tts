GUIDE_MD = """
This guide explains how to fine-tune the audio generation to get the best results.
---
#### 1. Text & Pre-processing (Input)

*   **Convert Numbers (🧮)**:
    *   Converts digits into words (e.g., "1998" -> "nineteen ninety-eight").
    *   *Recommended:* **Enabled**. If disabled, the model might ignore or mispronounce raw numbers.

*   **Split by (.!?) (✂️)**:
    *   Cuts long text into smaller chunks based on punctuation before sending it to the AI.
    *   *Enabled:* Faster generation and uses less RAM.
    *   *Disabled:* Processes the whole text at once. Better prosody flow but requires high VRAM and may crash on long texts.

#### 2. Emotion Settings (The "Soul")

*   **Emotion Source Modes**:
    1.  **Match Prompt Audio:** Uses the emotion found in the main Reference Voice.
    2.  **Use Emotion Ref:** You upload a *secondary* audio file just for the emotion (e.g., a crying sound), while keeping the voice identity of the main reference.
    3.  **Use Emotion Vector:** Manually adjust sliders (Joy, Anger, Sadness, etc.).
    4.  **Use Emotion Text:** Types a description (e.g., "Angry and shouting") to guide the AI. *Note: Requires Qwen model (heavier).*

*   **Emotion Strength/Weight**:
    *   How strongly the emotion overrides the default tone.
    *   **0.8 - 1.0:** Natural balance.
    *   **> 1.2:** Can lead to over-acting or audio artifacts.

#### 3. Sampling (The "Brain")

*   **Temperature (0.1 - 2.0)**:
    *   Controls randomness/creativity.
    *   **Low (0.1 - 0.5):** Stable, clear, monotone. Good for reading news.
    *   **Medium (0.7 - 0.9):** Natural intonation. **(Recommended)**
    *   **High (1.0+):** Very expressive but prone to mumbling or hallucinating words.

*   **Top P (0.0 - 1.0)** & **Top K (0 - 100)**:
    *   Filters out "bad" sound choices during generation.
    *   **Top P = 0.8, Top K = 30** is the standard for high-quality speech. Lowering these makes the voice more robotic/predictable.

*   **Do Sample**:
    *   *Checked:* Uses the parameters above (Human-like).
    *   *Unchecked:* Robotic/Deterministic mode (Greedy search).

#### 4. Timing & Structure

*   **Beam Search (Stability vs Emotion)**:
    *   Controls how carefully the AI plans the sentence.
    *   **1 (Recommended):** "Greedy/Sampling". Very emotional, natural, fast. *Cons:* May cut off audio abruptly.
    *   **3 - 5:** "Beam Search". Very stable, clear pronunciation, sentence always finishes. *Cons:* Robotic voice, 3x slower, consumes more VRAM.

*   **Interval Silence (ms)**:
    *   The gap inserted between sentences.
    *   **200ms:** Standard pause.
    *   **0ms (Crossfade Mode):** The end of one sentence blends into the start of the next. Use this for fast, continuous speech without awkward breaths.

*   **Max Text Tokens**:
    *   Limits how much text the AI "sees" at once per chunk.
    *   **~120:** Safe balance. Increasing this allows for longer sentences but increases the risk of the AI getting confused or repeating itself.

#### 5. Diffusion & Quality

*   **Diffusion Steps (10 - 50)**:
    *   How many times the vocoder "polishes" the raw audio.
    *   **10-15:** Fast, slightly gritty/static background.
    *   **25-30:** Good balance.
    *   **50:** Cleaner high frequencies and silence, but slower.

*   **CFG Scale (0.1 - 2.0)**:
    *   How hard the AI tries to mimic the *timbre* of the reference voice.
    *   **0.7:** Best match.
    *   **> 1.0:** Can make the voice sound metallic or oversaturated.

#### 6. Audio Post-Processing (The "Mix")

*These effects are applied **after** the AI generates the audio.*

*   **Speed Rate**:
    *   Changes the speed without changing the pitch.
    *   **1.0x:** Normal. **1.2x:** Snappy/YouTuber style.

*   **Pitch Shift**:
    *   Changes the tone (Semistones).
    *   **+2/-2:** Good for subtle gender/age tweaking.

*   **Normalize Volume**:
    *   Boosts quiet audio to a standard broadcast level (-0.5dB Peak) without clipping. Useful because AI output volume varies randomly.

*   **Auto-Clean Reference**:
    *   Applies a High-Pass filter to the *Input Voice* to remove low-frequency rumble/noise.
    *   *Disable this* if your reference voice sounds too thin or lacks bass.

"""
