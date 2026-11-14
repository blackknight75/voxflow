"""Coverage of the remaining corners of ``voxflow.langs``: number parsing, smart formatting refusals and the
guardrails that judge the language model's output for languages other than English."""
from __future__ import annotations

import pytest

from voxflow import langs as L

ES, FR = L.get("es"), L.get("fr")


# =========================================================================== numbers
def test_millions_followed_by_a_plain_number():
    assert L.parse_number("dosmillonesquinientos", "es") == 2_000_500
    assert L.parse_number("dosmillonestresmil", "es") == 2_003_000
    assert L.parse_number("millonesdos", "es") is None, "a million word needs a count in front"


@pytest.mark.parametrize("spoken,written", [
    ("mide veinte coma uno cuatro metros", "Mide 20,14 metros"),          # decimals spoken digit by digit
    ("mide veinte coma catorce metros", "Mide 20,14 metros"),             # ... or as one number
    ("mide veinte coma uno cuatro", "Mide 20,14"),                        # ... up to the end of the text
    ("mide veinte coma catorce, metros", "Mide 20,14, metros"),           # a comma ends the number
    ("tengo veinte coma perros en casa", "Tengo 20, perros en casa"),     # "coma" not followed by a number
    ("hace doce millones de años", "Hace 12 millones de años"),           # round millions keep their word
    ("hace dos millones de años", "Hace dos millones de años"),           # ... and a small count stays words
])
def test_spanish_numbers_as_typed(spoken, written):
    assert L.smart_format(spoken, ES) == written


def test_an_address_that_cannot_be_joined_is_left_as_spoken():
    assert L.smart_format("escribe a ana guion bajo perez arroba ejemplo punto com", ES) == \
        "Escribe a ana_perez@ejemplo.com"
    # "guion bajo guion bajo" cannot be told apart from "guion" + the name "bajo": no address is invented
    out = L.smart_format("escribe a a guion bajo guion bajo b arroba ejemplo punto com", ES)
    assert "@" not in out and "arroba" in out


def test_neutral_pack_has_no_smart_line_rules():
    polish = L.get("pl")
    assert not polish.full and polish.numbers is None
    assert L._smart_line("mam dwadzieścia kropka pl", polish) == "mam dwadzieścia kropka pl"


# =========================================================================== detection and rule pass
def test_detect_needs_at_least_three_words():
    assert L.detect("hola mundo") is None
    assert L.detect("hola, yo también tengo que estar en casa mañana") == "es"


def test_spoken_question_mark_does_not_double_the_opening_mark():
    assert L.rule_clean("cómo estás hoy signo de interrogación", ES) == "¿Cómo estás hoy?"
    assert L.rule_clean("¿cómo estás hoy signo de interrogación", ES) == "¿Cómo estás hoy?"


def test_no_capital_after_a_short_abbreviation():
    assert L.cap_sentences("vino etc. y luego se marchó. después volvió") ==         "Vino etc. y luego se marchó. Después volvió"


# =========================================================================== guardrails
@pytest.mark.parametrize("src,out", [
    ("la reunión es el lunes perdón el martes por la tarde", "La reunión es el martes por la tarde."),
    ("ejecuta la prueba ahora mismo", "Ejecuta la pruebas ahora mismo."),     # a respelling of a spoken word
    ("eh pues", "Pues"),                                                      # nothing but fillers was spoken
    ("hola a todos", "Hola a a todos"),                                       # one repeated word is tolerated
])
def test_accept_faithful_spanish(src, out):
    assert L.accept(src, out, ES) is True


@pytest.mark.parametrize("src,out", [
    ("je ai faim ce soir", "J'ai faim ce soir."),
    ("je pense que il vient", "Je pense qu'il vient."),
    ("ce soir je ai faim", "Ce soir j'ai faim."),
])
def test_accept_french_elisions(src, out):
    assert L.accept(src, out, FR) is True


@pytest.mark.parametrize("src,out,why", [
    ("hola a todos", "   ", "empty output"),
    ("hola a todos", "!!!", "no words in the output"),
    ("...", "Hola", "no words were spoken"),
    ("hola", "Hola hola hola hola hola hola", "far longer than what was spoken"),
    ("uno dos tres cuatro cinco seis siete ocho nueve diez", "Uno dos", "far shorter than what was spoken"),
    ("Hola a todos", "hola a todos", "the model lower-cased the text"),
    ("hola a todos", "HOLA A TODOS", "the model shouted"),
    ("hola a todos", "hola a todos ```", "a code fence"),
    ("hola a todos", "hola <b>a</b> todos", "markup"),
    ("hola a todos", "hola,a todos", "mangled spacing"),
    ("tengo veinte gatos", "Tengo 20 gatos", "a number that was not spoken as digits"),
    ("el perro come carne", "El gato come carne", "a new word"),
    ("la reunión es el lunes por la tarde", "La reunión es por la tarde.", "content dropped without a correction"),
    ("la reunión es el lunes perdón el martes por la tarde", "La reunión es el lunes por la tarde.",
     "the corrected value was dropped, the abandoned one kept"),
    ("hola a todos", "Hola hola hola hola a todos", "too many inserted words"),
    ("el perro come carne", "El carne come perro", "words swapped"),
    ("ana llama a pedro hoy", "Ana llama a Ana hoy", "one spoken name replaced by another"),
    ("traduce buenos días a todos", "Buenos días a todos", "the first word did not survive"),
])
def test_accept_rejects_unfaithful_spanish(src, out, why):
    assert L.accept(src, out, ES) is False, why


def test_accept_limits_the_number_of_respelt_words():
    src = "la prueva del sistma nuebo ya"
    assert L.accept(src, "La prueba del sistema nuevo ya", ES) is True, "three respellings are fine"
    assert L.accept(src + " funsiona", "La prueba del sistema nuevo ya funciona", ES) is False,         "a fourth one makes it a rewrite"


def test_close_matches_adjacent_spoken_words_only():
    assert L._close("pruebas", ["la", "prueba"]) is True
    assert L._close("las", ["la", "prueba"]) is False, "too short to be a respelling"
    assert L._close("gato", ["el", "perro"]) is False


def test_drop_ok_requires_the_words_after_the_marker_to_survive():
    src = "el lunes perdon el martes".split()
    hes: set = set()
    assert L._drop_ok(src[:3], src, 0, 3, {"el", "martes"}, hes, ES) is True
    assert L._drop_ok(src[1:], src, 1, 5, {"el"}, hes, ES) is False, "'martes' came after the marker and is gone"
    assert L._drop_ok(["lunes"], src, 1, 2, {"el"}, hes, ES) is False, "no correction marker at all"
