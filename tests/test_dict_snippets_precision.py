"""Dictionary and Snippets must never corrupt an ordinary sentence (review of fix/dict).

Precision over recall: a missed correction costs one edit, a changed sentence costs the user's trust. These are
the reviewer's examples plus a held-out set of normal sentences, with a realistic tech dictionary loaded, run
through the store directly and through the real controller pipeline (FakeTranscriber, real rule clean-up, and a
real ``Cleaner`` driven by a scripted stand-in for the language model).
"""
from __future__ import annotations

import tempfile
import threading

import pytest

from tests.controller_rig import make_rig
from voxflow import store
from voxflow.cleanup import Cleaner
from voxflow.fakes import FakeCleaner, FakeTranscriber

TECH_TERMS = [
    "Redis", "GitHub", "Kubernetes", "PyTorch", "Jenkins", "Angular", "Docker", "Grafana", "Postgres", "Kafka",
    "Looker", "Gusto", "Ashby", "Oisin", "Siobhan", "Cooper", "Spencer", "Monica", "Hunter", "Lattice",
    "Snowflake", "Svelte", "Elixir", "Mongo", "Notion", "Figma", "Linear", "Stripe", "Airflow", "Hubspot",
    "Mailchimp", "Salesforce", "Taylor", "Walker", "Archer", "Ansible", "Supabase", "Dagster", "Intercom",
    "Nguyen", "VoxFlow", "Tableau", "Plaid", "Mixpanel", "Zendesk", "Prometheus", "Cassandra",
]

# normal sentences, none of them about the terms above: every one must come through unchanged
HELD_OUT = [
    "Red is my colour.",
    "The flash drive is in the drawer.",
    "Look her up when you get to town.",
    "A gust of wind knocked the sign over.",
    "Ash be quiet for a minute.",
    "What angle are we taking on this?",
    "The copper wire is loose.",
    "He has always been a big spender.",
    "The soldiers wore leather jerkins.",
    "The hunter walked back to the cabin.",
    "The lattice on the fence needs paint.",
    "A snowflake landed on my nose.",
    "She looked svelte in that dress.",
    "They called it the elixir of life.",
    "He ate the whole cake with gusto.",
    "I would like a mango smoothie.",
    "The tailor fixed my jacket.",
    "The walker stopped at the gate.",
    "The archer aimed at the target.",
    "We need more air flow in this room.",
    "The sales force is growing fast.",
    "Is it in the box or on the shelf?",
    "That is just linear algebra.",
    "Stripes look good on you.",
    "The doctor will see you now.",
    "I have a notion that it will rain.",
    "Sigma is a Greek letter.",
    "Please call Monika tomorrow.",
    "The cooper made a new barrel.",
    "Let me get you a coffee.",
    "The whole nation was watching.",
    "Progress has been slow this week.",
    "The banana bread is in the oven.",
    "That sounds like a sensible plan.",
    "We need better governance here.",
    "Pack your suitcase tonight.",
    "Push the cart to the hub.",
    "Get hub caps for the car.",
    "The angular momentum is conserved.",
    "I mean it, she is in charge now.",
    "Did you get the email I sent?",
    "We played cards all night.",
    "The tablet is on the table.",
    "Mix the paint in the mix panel tray.",
    "Send the desk to the second floor.",
    "Stir the pot and add salt.",
    "Can we win this game?",
]


@pytest.fixture
def tech_dict():
    d = store.Dictionary(tempfile.mkdtemp())
    with d.batch():
        for t in TECH_TERMS:
            d.add(t)
    return d


@pytest.mark.parametrize("sentence", HELD_OUT)
def test_held_out_sentences_are_never_changed_by_the_dictionary(tech_dict, sentence):
    assert tech_dict.apply(sentence) == sentence


@pytest.mark.parametrize("sentence, wrong", [
    ("Red is my colour.", "Redis my colour."),
    ("Is in the garden.", "Oisin the garden."),
    ("Look her up.", "Looker up."),
    ("A gust of wind.", "A Gusto wind."),
    ("Ash be quiet.", "Ashby quiet."),
    ("The angle are wrong.", "The Angular wrong."),
    ("The copper pipe.", "The Cooper pipe."),
    ("A big spender.", "A big Spencer."),
    ("Leather jerkins.", "Leather Jenkins."),
])
def test_reviewer_examples_are_left_alone(sentence, wrong):
    d = store.Dictionary(tempfile.mkdtemp())
    for t in ("Redis", "Oisin", "Looker", "Gusto", "Ashby", "Angular", "Cooper", "Spencer", "Jenkins"):
        d.add(t)
    assert d.apply(sentence) == sentence != wrong


@pytest.mark.parametrize("term, heard", [
    ("Docker", "darker"), ("Docker", "decker"), ("Brandon", "branding"), ("Hunter", "hunted"),
    ("Parker", "parked"), ("Spencer", "spender"), ("Kafka", "coffee"), ("Redis", "radish"),
    ("Jenkins", "jerkins"), ("Notion", "nation"), ("Figma", "sigma"), ("Daniel", "denial"),
    ("Rachel", "racial"), ("Sophie", "sofa"), ("Nadia", "media"), ("Vercel", "vessel"),
    ("Laravel", "unravel"), ("Grafana", "banana"), ("Ansible", "sensible"), ("Kotlin", "codling"),
    ("Svelte", "svelte"), ("Kubernetes", "governance"), ("Postgres", "progress"), ("Supabase", "suitcase"),
    ("Siobhan", "shove on"), ("Oisin", "is in"), ("Gusto", "gust of"), ("Redis", "red is"),
    ("Looker", "look her"), ("Ashby", "ash be"), ("Elixir", "elixir"), ("Ronan", "Ronin"), ("Colm", "calm"),
    ("Tableau", "tablet"), ("Kafka", "calf car"), ("Hubspot", "hub spot"), ("Zendesk", "send desk"),
    ("Mixpanel", "mix panel"), ("Cassandra", "cassette"), ("Prometheus", "promote this"),
    ("Dagster", "dag stir"), ("Angular", "angle are"), ("Plaid", "played"), ("Lattice", "lettuce"),
    ("Mongo", "mango"), ("Mongo", "among go"), ("Monica", "Monika"), ("Salesforce", "sales force"),
    ("Mailchimp", "mail chimp"), ("Airflow", "air flow"), ("Snowflake", "snowflake"), ("Hunter", "hunter"),
])
def test_real_words_never_become_a_term_without_an_alias(term, heard):
    d = store.Dictionary(tempfile.mkdtemp())
    d.add(term)
    s = f"I said {heard} today."
    assert d.apply(s) == s


def test_an_explicit_alias_is_the_way_to_turn_real_words_into_a_term():
    d = store.Dictionary(tempfile.mkdtemp())
    d.add("Redis", ["red is"])
    d.add("PyTorch", ["pie torch"])
    assert d.apply("We cache it in red is.") == "We cache it in Redis."
    assert d.apply("Train it in pie torch.") == "Train it in PyTorch."


def test_misspellings_that_are_not_words_are_still_corrected(tech_dict):
    for heard, fixed in (("We deploy on Kubernetis.", "We deploy on Kubernetes."),
                         ("Open the Graphana dashboard.", "Open the Grafana dashboard."),
                         ("Kubernetis's scheduler.", "Kubernetes's scheduler."),
                         ("I dictate with vox flow.", "I dictate with VoxFlow."),
                         ("It runs in py torch.", "It runs in PyTorch."),
                         ("Push it to get hub.", "Push it to get hub."),           # "get", "hub": words
                         ("Push it to git hub.", "Push it to GitHub."),            # "git" is not
                         ("We use kubernetes.", "We use Kubernetes.")):            # not a word: case fixed
        assert tech_dict.apply(heard) == fixed


def test_a_term_that_is_an_ordinary_word_keeps_the_sentence_as_written(tech_dict):
    tech_dict.add("OneDrive")
    assert tech_dict.apply("I only have one drive.") == "I only have one drive."  # "one", "drive": words
    assert tech_dict.apply("Save it to One Drive.") == "Save it to OneDrive."      # with the term's capitals
    assert tech_dict.apply("Use Linear for tickets, not linear search.") == \
        "Use Linear for tickets, not linear search."


def test_other_languages_are_not_fuzzy_corrected(tech_dict):
    for s in ("Wir treffen uns morgen früh im Büro mit Kubernetis.", "Nous allons au marché demain matin.",
              "Vamos a la playa mañana por la tarde."):
        assert tech_dict.apply(s) == s


def test_without_the_word_list_fuzzy_correction_is_off(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "_WORDLISTS", None)
    monkeypatch.setattr(store, "_DATA_DIR", tmp_path / "missing")
    d = store.Dictionary(tempfile.mkdtemp())
    d.add("Kubernetes", ["cube net ease"])
    assert d.apply("Kubernetis and cube net ease.") == "Kubernetis and Kubernetes."   # aliases still work
    monkeypatch.setattr(store, "_WORDLISTS", None)


def test_the_word_list_is_small_and_knows_ordinary_words():
    words, names = store._wordlists()
    assert 50_000 < len(words) < 80_000 and len(names) < 15_000
    for w in ("copper", "spender", "jerkins", "colour", "color", "lettuce", "governance"):
        assert w in words
    for name in ("jenkins", "redis", "monica", "kubernetes"):
        assert name not in words
    size = sum((store._DATA_DIR / f).stat().st_size for f in ("english_words.txt.gz", "english_names.txt.gz"))
    assert size < 300_000


# =========================================================================== snippets
@pytest.fixture
def sn():
    s = store.Snippets(tempfile.mkdtemp())
    s.add("my address", "1 Main St, Springfield")
    s.add("sign off", "Best,\nDan")
    s.add("email", "dan@example.com")
    s.add("thanks", "Thank you very much")
    s.add("meeting link", "https://meet.example/abc")
    s.add("meine adresse", "Hauptstraße 1")
    s.add("mijn adres", "Dorpsstraat 1")
    s.add("ma signature", "Cordialement, Dan")
    return s


@pytest.mark.parametrize("sentence", [
    "I will sign off on the budget.",
    "Send me an email tomorrow.",
    "Did you get the email?",
    "Thanks for the help.",
    "Thanks, see you soon.",
    "I need to change my address next week.",
    "Please send it to my address.",
    "Can you share the meeting link with Bob?",
    "Ich habe meine Adresse geändert.",
    "Thanks, sign off.",
])
def test_a_trigger_inside_an_ordinary_sentence_is_left_alone(sn, sentence):
    assert sn.expand(sentence) == sentence
    assert not sn.matches(sentence)


@pytest.mark.parametrize("spoken, pasted", [
    ("Insert my address, please.", "1 Main St, Springfield, please."),
    ("Please insert my address.", "1 Main St, Springfield"),
    ("Send it to, insert my address.", "Send it to, 1 Main St, Springfield."),
    ("Thanks for that. Paste sign off.", "Thanks for that. Best,\nDan"),
    ("Type the email here.", "dan@example.com here."),
    ("Insert E-mail.", "dan@example.com"),
    ("E-mail.", "dan@example.com"),
    ("Um, sign off.", "Best,\nDan"),
    ("\"Thanks.\"", "Thank you very much"),
    ("Insert my address and paste the meeting link.", "1 Main St, Springfield and https://meet.example/abc."),
    ("Bitte meine Adresse einfügen.", "Bitte Hauptstraße 1."),
    ("Bitte füge meine Adresse ein.", "Hauptstraße 1"),             # the whole dictation asks for it
    ("Dann füge meine Adresse ein.", "Dann Hauptstraße 1."),
    ("Voeg mijn adres in, alsjeblieft.", "Dorpsstraat 1, alsjeblieft."),
    ("Insère ma signature.", "Cordialement, Dan"),
    ("Merci. Colle ma signature.", "Merci. Cordialement, Dan"),
    ("Pega my address, por favor.", "1 Main St, Springfield, por favor."),
    ("Incolla my address.", "1 Main St, Springfield"),
    ("Cole my address.", "1 Main St, Springfield"),
])
def test_a_snippet_fires_when_asked_for_and_the_lead_word_goes(sn, spoken, pasted):
    assert sn.expand(spoken) == pasted
    assert sn.matches(spoken)


def test_the_lead_word_must_come_right_before_the_trigger(sn):
    assert sn.expand("Insert a note about my address.") == "Insert a note about my address."
    assert sn.expand("Insert my addresses here.") == "Insert my addresses here."          # whole words only


def test_hyphen_variants_of_one_word_trigger(sn):
    s = store.Snippets(tempfile.mkdtemp())
    s.add("e-mail", "E")
    s.add("wifi password", "W")
    for spoken in ("Email.", "E-mail.", "e mail", "Insert e-mail."):
        assert s.expand(spoken) == "E", spoken
    for spoken in ("Wi-Fi password.", "WiFi password", "Insert wi-fi password."):
        assert s.expand(spoken) == "W", spoken


# =========================================================================== through the controller
def _rig(tmp_path, qtbot, monkeypatch, cleaner):
    return make_rig(tmp_path, qtbot, monkeypatch, asr=FakeTranscriber(text="", delay_s=0.0), cleaner=cleaner)


def _say(rig, spoken: str) -> str:
    before = len(rig.injector.pasted)
    rig.dictate(spoken)
    assert rig.c.state != "error", rig.c.message
    assert len(rig.injector.pasted) == before + 1, "nothing was pasted"
    return rig.injector.pasted[-1]


def test_held_out_sentences_come_through_the_pipeline_unchanged(tmp_path, qtbot, monkeypatch):
    r = _rig(tmp_path, qtbot, monkeypatch, FakeCleaner(delay_s=0.0))
    try:
        with r.c.dictionary.batch():
            for t in TECH_TERMS:
                r.c.dictionary.add(t)
        for t in ("sign off", "email", "thanks", "my address"):
            r.c.snippets.add(t, "<" + t.upper() + ">")
        rules = Cleaner(use_llm=False)
        rules.load()
        for s in HELD_OUT + ["I will sign off on the budget.", "Send me an email tomorrow.", "Thanks for the help."]:
            assert _say(r, s) == rules.clean(s, language="auto", smart=True), s
    finally:
        r.close()


class ScriptedLLM:
    """Stands in for llama_cpp.Llama inside a real ``Cleaner``: replies come from ``script(user_text)``."""

    def __init__(self, script):
        self.script = script
        self.calls: list[list[dict]] = []

    def tokenize(self, data: bytes, add_bos=True):
        return list(range(max(1, len(data.split()))))

    def create_chat_completion(self, messages, temperature, max_tokens, stream, **kw):
        self.calls.append(messages)
        reply = self.script(messages[-1]["content"])
        return iter([{"choices": [{"delta": {"content": reply}}]}, {"choices": [{"delta": {}}]}])


class CleanerWithLLM(Cleaner):
    """The real Cleaner (rules, ``_needs_llm``, prompt, guardrails); only the model itself is scripted."""

    def __init__(self, script):
        super().__init__(timeout_s=4.0)
        self.fake = ScriptedLLM(script)
        self.loaded = threading.Event()

    def load(self) -> None:
        self._llm, self._ready, self.backend = self.fake, True, "cuda"
        self.loaded.set()


def test_language_model_gets_the_dictionary_and_snippets_survive_its_rewording(tmp_path, qtbot, monkeypatch):
    rewrites = {
        # the model only tidies; the rules already made the text sentence-like
        "Insert my address, please.": "Insert my address, please.",
        # the model "improves" the command so the trigger is gone: the controller falls back to the transcript
        "So, insert my address.": "Please add the address.",
        "We use Kubernetis and Graphana.": "We use Kubernetes and Grafana.",
        "I will sign off on the budget tomorrow.": "I will sign off on the budget tomorrow.",
    }
    c = CleanerWithLLM(lambda user: rewrites.get(user, user))
    r = _rig(tmp_path, qtbot, monkeypatch, c)
    try:
        r.c.dictionary.add("Kubernetes")
        r.c.dictionary.add("Grafana")
        r.c.snippets.add("my address", "1 Main St")
        r.c.snippets.add("sign off", "Best,\nDan")

        assert _say(r, "um insert my address, please") == "1 Main St, please."
        assert c.last_info.get("llm") is True, "this dictation went through the language model"
        system = c.fake.calls[-1][0]["content"]
        assert "Kubernetes" in system and "Grafana" in system, "the dictionary terms are in the model's prompt"

        # the model reworded the trigger away: the controller falls back to the rule-cleaned transcript
        assert _say(r, "uh so, insert my address") == "So, 1 Main St."
        assert _say(r, "um we use kubernetis and graphana") == "We use Kubernetes and Grafana."
        assert _say(r, "um I will sign off on the budget tomorrow") == "I will sign off on the budget tomorrow."
    finally:
        r.close()
