"""
Comprehensive Banglish vocabulary hints, code-switching patterns, and
known Whisper mishearing mappings for Bengali phonemes.

Used by:
  - transcriber.py  → builds the Whisper ``initial_prompt`` so the decoder
                       expects Banglish tokens.
  - interpreter.py  → injected into the Claude system prompt so it knows
                       which Whisper mistakes to look for.
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
# 1.  CORE BANGLISH VOCABULARY
#     Common Bengali words in Roman script, grouped by part of speech /
#     semantic category.  Each entry is the most common romanisation;
#     variant spellings are listed where they diverge significantly.
# ═══════════════════════════════════════════════════════════════════════════

PRONOUNS = [
    "ami", "amra", "tumi", "tomra", "tui", "tora",
    "apni", "apnara", "she", "shey", "tini", "ora", "tara",
    "amar", "amader", "tomar", "tomader", "tar", "tader",
    "oder", "oder", "apnar", "apnader",
    "nijer", "nijeke", "amake", "tomake", "take", "oke",
]

VERBS_COMMON = [
    # being / existence
    "achi", "acho", "ache", "achen", "chilo", "chilen", "thakbo",
    "hoy", "hobe", "hoyeche", "hoye", "holo", "hocche",
    # doing
    "kori", "koro", "kore", "koren", "korbo", "korbe", "korche",
    "korchilo", "korechilo", "korte",
    # going / coming
    "jai", "jao", "jay", "jabe", "jabo", "jacchi", "jacche",
    "gelo", "geche", "gechilo", "jete",
    "ashi", "asho", "ashe", "ashbe", "ashbo", "ashche", "elo", "eche",
    # saying / telling
    "boli", "bolo", "bole", "bolbo", "bolbe", "bolche", "bolchilo",
    # seeing / knowing
    "dekhi", "dekho", "dekhe", "dekhbe", "dekhbo", "dekhche",
    "jani", "jano", "jane", "janbe", "janbo", "janche",
    # eating / drinking
    "khai", "khao", "khay", "khabe", "khabo", "khacche",
    # giving / taking
    "dei", "dao", "dey", "dibe", "dibo", "dicche",
    "nei", "nao", "ney", "nibe", "nebo", "nicche",
    # wanting / needing
    "chai", "chao", "chay", "chaibe", "chaibo",
    "lagbe", "lagche", "dorkar",
    # understanding
    "bujhi", "bujho", "bujhe", "bujhbe", "bujhbo", "bujhche",
    # writing / reading
    "likhi", "likho", "likhe", "likhbo", "likhbe",
    "pori", "poro", "pore", "porbo", "porbe",
    # thinking
    "bhabi", "bhabo", "bhabe", "bhabchi", "bhabbo",
    # speaking
    "koi", "kotha", "bolchi",
    # sleeping / sitting / standing
    "ghumao", "ghumacchi", "ghum",
    "boshi", "bosho", "boshe", "boshbo",
    "darao", "dariye", "darabo",
    # loving / liking
    "bhalobashi", "bhalobaso", "bhalobashbe", "bhalo lage",
    # miscellaneous
    "pari", "paro", "pare", "parbe", "parbo",
    "rakhbo", "rakhbe", "rakhe", "rakho",
    "shuru", "shesh",
]

QUESTION_WORDS = [
    "ki", "ke", "keno", "kothay", "kokhon", "kivabe", "koto",
    "kisher", "kar", "kake", "kiser", "kemne", "kemon",
]

CONJUNCTIONS_PARTICLES = [
    "ar", "o", "ba", "ebong", "kintu", "tobe", "tahole",
    "jodi", "nahole", "tai", "karon", "jekhane", "sekhane",
    "jokhon", "tokhon", "jeno", "naki", "mane", "matlab",
    "akhon", "ekhon", "tokhon", "pore", "age",
]

ADJECTIVES_ADVERBS = [
    "bhalo", "kharap", "boro", "choto", "sundor", "onek",
    "kom", "beshi", "shob", "shobai", "shudhu", "ektu",
    "thik", "ghoton", "nishchit", "abar", "ebar", "shotti",
    "puro", "dhoroner", "alada", "notun", "purano",
    "dure", "kache", "upore", "niche", "baire", "bhitore",
    "aste", "taratari", "jore",
]

NOUNS_COMMON = [
    "manush", "lok", "chhele", "meye", "bhai", "bon", "baba",
    "ma", "ammu", "abbu", "chacha", "mama", "khala", "fufu",
    "bondhu", "dost", "bari", "ghor", "rasta", "gaari",
    "paani", "khabar", "bhat", "mach", "torkari", "cha",
    "taka", "poisha", "kaj", "chakri", "school", "college",
    "desh", "shohor", "gram", "din", "raat", "shokal",
    "bikel", "shondha", "shomoy", "jibon", "mrittu",
    "kotha", "golpo", "boi", "kagoj", "kalam",
    "phone", "computer", "message",
    "dokan", "bazaar", "hospital", "doctor", "oshudh",
]

GREETINGS_EXPRESSIONS = [
    "assalamu alaikum", "walaikum assalam",
    "kemon acho", "kemon achen", "bhalo achi",
    "ki khobor", "ki holo", "ki korcho", "ki korchen",
    "accha", "thik ache", "haan", "na", "jee",
    "dhonnobad", "shukriya", "please", "sorry", "thank you",
    "mashallah", "inshallah", "alhamdulillah", "subhanallah",
    "ei je", "oi je", "are baba", "ore baba",
    "ki je kori", "ki bolbo", "arey", "orey", "hey",
    "ekdom", "bilkul", "shotti", "sotti", "obviously",
]

NUMBERS = [
    "ek", "dui", "tin", "char", "pach", "chhoy", "shat", "aat",
    "noy", "dosh", "egaro", "baro", "tero", "choddo", "ponero",
    "shollo", "shotero", "aaro", "unish", "bish",
    "tish", "chollish", "ponchash", "shaath", "sottor",
    "ashi", "nobboi", "ekso",
]


# ═══════════════════════════════════════════════════════════════════════════
# 2.  CODE-SWITCHING PATTERNS
#     Banglish speakers insert English words/phrases in predictable slots.
#     These examples teach both Whisper and Claude what to expect.
# ═══════════════════════════════════════════════════════════════════════════

CODE_SWITCH_PATTERNS = [
    # English verbs with Bangla inflection
    "call korbo", "check koro", "start kore dao",
    "send kore dibo", "try korchi", "manage korte parbo",
    "download korte hobe", "upload kore dao",
    "install koro", "update kore nao", "fix korbo",
    "order diyechi", "cancel koro", "book kore rakhbo",
    "search koro", "share koro", "join korbo",

    # English nouns in Bangla sentences
    "meeting e jabo", "office theke", "phone e kotha bolo",
    "email ta pathao", "message ta dekho", "file ta open koro",
    "bus e uthe", "train dhorte hobe", "ticket katao",
    "class e achi", "exam er jonno", "project ta shesh koro",

    # English adjectives/adverbs with Bangla
    "important jinish", "serious bishoy", "ektu busy achi",
    "onek boring", "sure na", "ready achi", "already hoye geche",
    "actually ami", "basically oita", "obviously bujhte parcho",
    "definitely korbo", "probably jabo na",

    # Whole English clauses mid-sentence
    "ami think kori je", "tumi sure na?",
    "ami at least try korbo", "oita kind of different",
    "eta too much hoye jacche", "sheta make sense kore",
]


# ═══════════════════════════════════════════════════════════════════════════
# 3.  WHISPER MISHEARING MAP
#     Bengali has aspirated stops (kh, gh, chh, jh, th, dh, ph, bh),
#     retroflex consonants, and nasal sounds that Whisper's English-biased
#     decoder frequently mis-maps.  Each entry is:
#         "what Whisper often produces" → ["what was likely said", ...]
# ═══════════════════════════════════════════════════════════════════════════

WHISPER_MISHEARINGS: dict[str, list[str]] = {
    # ── Aspirated stops ──────────────────────────────────────────────
    # kh (খ)
    "car":      ["khar", "kharap"],
    "come":     ["khabar", "khao"],
    "cow":      ["khao"],
    "con":      ["khon", "kokhon"],

    # gh (ঘ)
    "go":       ["ghor", "ghum"],
    "gore":     ["ghor e"],
    "goom":     ["ghum"],
    "guru":     ["ghure"],

    # chh (ছ)
    "chair":    ["chhele"],
    "child":    ["chhoto"],
    "chosen":   ["chhobi"],

    # jh (ঝ)
    "jaw":      ["jhal", "jhol"],
    "Joel":     ["jhol"],
    "jar":      ["jhar"],

    # th (থ / ঠ)  — Whisper often drops aspiration
    "tick":     ["thik"],
    "take":     ["thake", "thako"],
    "talk":     ["taka"],

    # dh (ধ / ঢ)
    "door":     ["dhor", "dhore"],
    "done":     ["dhon", "dhonnobad"],
    "donor":    ["dhoroner"],

    # ph (ফ)  — Whisper tends to hear "f" or "p"
    "phone":    ["phone", "phon"],  # sometimes correct, sometimes not
    "full":     ["phul"],
    "fool":     ["phul"],
    "fun":      ["phan"],

    # bh (ভ)
    "buy":      ["bhai"],
    "bye":      ["bhai"],
    "bar":      ["bhar", "bhari"],
    "bible":    ["bhabi"],
    "bob":      ["bhab", "bhabo"],
    "ball":     ["bhalo", "bhalobashi"],
    "below":    ["bhalo"],
    "bow":      ["bhalo", "bhalobashi"],
    "bah":      ["bhai", "bhabna"],

    # ── Sibilants sh (শ/ষ) vs s (স) ─────────────────────────────────
    "show":     ["shob", "shokal"],
    "shore":    ["shohor"],
    "sob":      ["shob"],
    "sub":      ["shob", "shobai"],
    "should":   ["shudhu"],
    "shoe":     ["shuru"],
    "sue":      ["shuru"],
    "shower":   ["shohor"],
    "sugar":    ["shukriya"],
    "sundry":   ["sundor"],
    "sure":     ["shure"],
    "sorry":    ["shotti"],
    "saw":      ["shob"],

    # ── Retroflex vs dental ──────────────────────────────────────────
    # ড vs D, ট vs T — Whisper hears English "d/t"
    "dock":     ["dosh", "dorkar"],
    "done":     ["dosh"],
    "tall":     ["tala", "taka"],

    # ── Nasal ং / ঞ / ণ / ন ─────────────────────────────────────────
    "among":    ["amongo"],  # trailing -ng
    "bang":     ["Bangla"],
    "bone":     ["bon"],
    "money":    ["mane"],
    "man":      ["manush", "mane"],
    "month":    ["mondo"],

    # ── Vowel confusions (Bengali has more vowel shades) ─────────────
    "a":        ["e", "ae"],  # æ vs এ
    "egg":      ["ek"],
    "eight":    ["aat"],
    "auto":     ["auto", "oto"],
    "all":      ["aal", "ar"],
    "one":      ["onek"],
    "own":      ["onek"],
    "only":     ["onek"],
    "or":       ["ar"],
    "oar":      ["ar"],
    "echo":     ["ektuo", "ektu"],

    # ── Common full-word swaps ───────────────────────────────────────
    "call":     ["kol", "kal"],       # kol (factory) / kal (tomorrow/yesterday)
    "key":      ["ki"],               # ki = what
    "color":    ["kolar"],            # kolar = banana
    "she":      ["shey"],             # shey = he/she (gender-neutral in Bangla)
    "he":       ["hae", "hoy"],
    "I":        ["ami"],
    "me":       ["ami", "amake"],
    "my":       ["amar"],
    "we":       ["amra"],
    "day":      ["dei", "dao"],       # dei/dao = give
    "die":      ["dai"],
    "the":      ["ta", "ti"],
    "no":       ["na"],
    "not":      ["na", "noy"],
    "can":      ["kan"],              # kan = ear
    "much":     ["mach"],             # mach = fish
    "pardon":   ["pari", "parbo"],
    "pass":     ["pash", "pari"],
    "rash":     ["rasta"],
    "caught":   ["koto"],             # koto = how much
    "cold":     ["kol"],
    "good":     ["ghor"],
    "nah":      ["na"],
    "yeah":     ["hae", "jee"],
    "okay":     ["accha", "thik ache"],
    "to me":    ["tumi"],
    "tomb":     ["tumi"],
}


# ═══════════════════════════════════════════════════════════════════════════
# 4.  BUILT STRINGS — ready to drop into Whisper / Claude prompts
# ═══════════════════════════════════════════════════════════════════════════

def _flatten_vocab() -> list[str]:
    """Merge all word lists into a single de-duplicated list."""
    seen: set[str] = set()
    out: list[str] = []
    for pool in (
        PRONOUNS, VERBS_COMMON, QUESTION_WORDS, CONJUNCTIONS_PARTICLES,
        ADJECTIVES_ADVERBS, NOUNS_COMMON, GREETINGS_EXPRESSIONS, NUMBERS,
    ):
        for w in pool:
            low = w.lower()
            if low not in seen:
                seen.add(low)
                out.append(w)
    return out


ALL_VOCAB: list[str] = _flatten_vocab()


# ── Whisper initial_prompt ────────────────────────────────────────────────
# Whisper's decoder uses the initial_prompt as conditioning context.
# We pack in as many representative Banglish tokens as feasible (the model
# truncates to its context length automatically, so overshoot is fine).

def build_whisper_prompt() -> str:
    """
    Build a rich initial_prompt string for ``whisper.transcribe()``.

    Combines:
      • A short description of Banglish
      • A dense sample of vocabulary tokens
      • A handful of code-switching example sentences
    """
    # Take a generous slice of vocab (Whisper will truncate if needed).
    vocab_sample = ", ".join(ALL_VOCAB[:180])

    pattern_sample = ". ".join(CODE_SWITCH_PATTERNS[:20])

    return (
        "This is a conversation in Banglish — Bengali (Bangla) spoken and "
        "written in English/Latin letters, frequently code-switching with "
        "English words and phrases mid-sentence. "
        "Bengali has aspirated consonants (kh, gh, chh, jh, th, dh, ph, bh) "
        "and retroflex sounds that may resemble but are distinct from English "
        "phonemes.\n\n"
        f"Common Banglish words: {vocab_sample}.\n\n"
        f"Example code-switching: {pattern_sample}.\n\n"
        "The speaker will freely mix Bengali and English. Transcribe Bengali "
        "words in their Roman/Latin transliteration, not in Bengali script."
    )


WHISPER_PROMPT: str = build_whisper_prompt()


# ── Claude interpreter reference ──────────────────────────────────────────

def build_claude_reference() -> str:
    """
    Build a reference block for the Claude system prompt that lists
    known Whisper mishearings and common Banglish patterns.
    """
    lines: list[str] = []

    lines.append("## Known Whisper mishearings of Bengali phonemes\n")
    lines.append(
        "Bengali has aspirated stops (kh খ, gh ঘ, chh ছ, jh ঝ, th থ/ঠ, "
        "dh ধ/ঢ, ph ফ, bh ভ), retroflexes (ট, ড), and sibilant "
        "distinctions (sh শ/ষ vs s স) that Whisper's English-biased "
        "decoder often maps to the wrong English word. Common swaps:\n"
    )
    for eng, bangla_opts in WHISPER_MISHEARINGS.items():
        lines.append(f"  • \"{eng}\" → {', '.join(bangla_opts)}")

    lines.append("\n## Common Banglish code-switching patterns\n")
    lines.append(
        "Speakers frequently embed English nouns, verbs, adjectives, or "
        "entire clauses inside Bengali sentence structure:\n"
    )
    for pat in CODE_SWITCH_PATTERNS[:15]:
        lines.append(f"  • {pat}")

    lines.append("\n## High-frequency Banglish vocabulary\n")
    lines.append(
        "These are the most common Bengali words you will encounter in "
        "Roman script (pronouns, verbs, question words, particles, etc.):\n"
    )
    # Chunk into readable rows of ~10 words.
    chunk_size = 10
    for i in range(0, min(len(ALL_VOCAB), 120), chunk_size):
        chunk = ALL_VOCAB[i : i + chunk_size]
        lines.append("  " + ", ".join(chunk))

    return "\n".join(lines)


CLAUDE_REFERENCE: str = build_claude_reference()
