import pytest

from app.modules.red_flags import check, check_slots

POSITIVE = [
    ("papa behosh ho gaye", "unresponsive"),
    ("mummy behosh hain aur uth nahi rahi", "unresponsive"),
    ("dada ji ko hosh nahi hai", "unresponsive"),
    ("he is unconscious", "unresponsive"),
    ("patient not responding to anything", "unresponsive"),
    ("bhaiya behooosh ho gaye", "unresponsive"),          # repeated letters
    ("papa bahosh ho gaye hain", "unresponsive"),          # spelling variant
    ("पापा बेहोश हो गए", "unresponsive"),
    ("saans nahi aa rahi", "breathing"),
    ("saaans nahi aa rahi mummy ko", "breathing"),
    ("saans phool rahi hai", "breathing"),
    ("saans lene mein takleef ho rahi hai", "breathing"),
    ("sans lene me dikkat hai bahut", "breathing"),
    ("he is breathless", "breathing"),
    ("I can't breathe", "breathing"),
    ("difficulty breathing since night", "breathing"),
    ("सांस नहीं आ रही है", "breathing"),
    ("दम घुट रहा है", "breathing"),
    ("hoth neele ho gaye hain", "blue_lips"),
    ("his lips are blue", "blue_lips"),
    ("होंठ नीले पड़ गए", "blue_lips"),
    ("seene mein dard hai aur paseena aa raha hai", "chest_pain_combo"),
    ("chest pain and left arm pain", "chest_pain_combo"),
    ("chhati me dard aur jabda dard", "chest_pain_combo"),
    ("सीने में दर्द और पसीना", "chest_pain_combo"),
    ("papa ko heart attack jaisa lag raha hai", "chest_pain_combo"),
    ("muh tedha ho gaya hai", "stroke"),
    ("face drooping and slurred speech", "stroke"),
    ("ek taraf kamzori hai", "stroke"),
    ("bol nahi pa rahe", "stroke"),
    ("लकवा मार गया लगता है", "stroke"),
    ("daura pad raha hai", "seizure"),
    ("bachche ko jhatke aa rahe hain", "seizure"),
    ("seizure ho raha hai", "seizure"),
    ("मिर्गी का दौरा", "seizure"),
    ("khoon band nahi ho raha", "severe_bleeding"),
    ("heavy bleeding from the leg", "severe_bleeding"),
    ("bahut khoon beh raha hai", "severe_bleeding"),
    ("sir pe chot lagi aur ulti ho rahi hai", "head_injury"),
    ("head injury, he is confused", "head_injury"),
    ("galti se dawai kha li", "poisoning"),
    ("usne zeher kha liya", "poisoning"),
    ("overdose of pills", "poisoning"),
    ("saanp ne kata hai", "poisoning"),
    ("pregnant hai aur bahut khoon aa raha hai", "pregnancy_emergency"),
    ("garbhvati hain aur pani gir gaya", "pregnancy_emergency"),
    ("10 din ka bachcha hai bukhar hai", "infant_fever"),
    ("newborn ko bukhar hai", "infant_fever"),
    ("2 mahine ka baby hai doodh nahi pi raha", "infant_fever"),
    ("bahut jal gaya hai chehra", "severe_burn"),
    ("khud ko nuksan pahunchana chahta hai", "suicidal"),
    ("he wants to commit suicide", "suicidal"),
    ("papa behosh hain, bukhar nahi hai", "unresponsive"),   # negator belongs to another clause
    ("chest pain ho raha hai papa ko", "chest_pain"),         # organisers' multi-turn example MT002
    ("seene mein halka dard hai", "chest_pain"),              # chest pain alone is a red flag in the organisers' data
    ("wo achanak gir gaye response nahi de rahe", "unresponsive"),
    ("He suddenly collapsed and is not responding", "unresponsive"),
    ("accident ke baad bleeding ruk nahi rahi", "severe_bleeding"),
    ("Heavy bleeding that won't stop after an accident", "severe_bleeding"),
    ("एक्सीडेंट के बाद खून नहीं रुक रहा", "severe_bleeding"),
    ("वो अचानक गिर गए और जवाब नहीं दे रहे", "unresponsive"),
    ("patient is severely dehydrated", "severe_dehydration"),
]

NEGATIVE = [
    "papa ko do din se bukhar hai aur khansi bhi",
    "saans mein koi dikkat nahi hai",
    "saans lene mein koi dikkat nahi",
    "breathing is normal",
    "behosh nahi hue hain",
    "he is not unconscious",
    "paseena bahut aa raha hai garmi se",           # sweat alone
    "5 saal ka bachcha hai bukhar hai",             # not an infant
    "6 mahine ka bachcha hai bukhar hai",           # >= 3 months
    "sir dard hai halka",
    "khoon ki jaanch karwani hai",
    "mummy ko zukam hai",
    "hoth sukh gaye hain",
    "injection lagwana hai",
    "मुझे हल्का बुखार है",
    "सांस की कोई दिक्कत नहीं है",
    "no difficulty with anything, all normal",
]


@pytest.mark.parametrize("text,flag", POSITIVE)
def test_positive(text, flag):
    fired, flags = check(text)
    assert fired and flag in flags, (text, flags)


@pytest.mark.parametrize("text", NEGATIVE)
def test_negative(text):
    fired, flags = check(text)
    assert not fired, (text, flags)


def test_history_is_scanned():
    fired, flags = check("haan", [{"role": "user", "content": "papa behosh hain"}])
    assert fired and "unresponsive" in flags


def test_slot_gate():
    assert check_slots({"breathing_difficulty": "yes"}) == ["breathing"]
    assert check_slots({"breathing_difficulty": "no", "chest_pain": "unknown"}) == []


def test_gate_agrees_with_every_organiser_label():
    """The rules-first gate fires on exactly the organisers' emergency requests (340 labelled rows)."""
    from app.data_loader import load_cases, load_taxonomy
    em = load_taxonomy().emergency_label
    bad = [(c.text, c.care_type) for c in load_cases() if check(c.text)[0] != (c.care_type == em)]
    assert not bad, bad


def test_vocabulary_mapping():
    from app.data_loader import load_taxonomy
    from app.modules.red_flags import VOCAB, to_vocab
    vocab = set(load_taxonomy().red_flag_vocab)
    assert set(VOCAB.values()) <= vocab
    assert to_vocab(["breathing", "blue_lips", "unresponsive"]) == ["breathlessness", "unconsciousness"]
