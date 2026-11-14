"""The non-English dictation pipeline (round 3): ``voxflow.langs`` and its use in ``voxflow.cleanup``,
``voxflow.command``, ``voxflow.fakes`` and the Controller's static helpers.

No real model is loaded: the language model is a scripted stand-in.
Languages with their own rules ("full"): es, fr, de, it, pt, nl. Everything else is "basic".
"""
from __future__ import annotations

import pytest

from voxflow import cleanup, command, fakes, langs
from voxflow.cleanup import Cleaner, apply_tone, clean_foreign, is_english, resolve_language

FULL = ("es", "fr", "de", "it", "pt", "nl")


def bug(bug_id: str):
    return pytest.mark.xfail(reason=f"open bug {bug_id}", strict=False)


@pytest.fixture(scope="module")
def rules() -> Cleaner:
    c = Cleaner(use_llm=False)
    c.load()
    return c


class FakeLLM:
    """Stands in for llama_cpp.Llama: one scripted reply (or exception), streamed word by word."""

    def __init__(self, reply):
        self.reply = reply
        self.calls: list[list[dict]] = []

    def tokenize(self, data: bytes, add_bos=True):
        return list(range(max(1, len(data.split()))))

    def create_chat_completion(self, messages, temperature, max_tokens, stream, **kw):
        self.calls.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        words = self.reply.split(" ")

        class Stream:
            def __iter__(self):
                for w in words:
                    yield {"choices": [{"delta": {"content": w + " "}}]}
                yield {"choices": [{"delta": {}}]}

            def close(self):
                pass

        return Stream()


def with_llm(reply) -> tuple[Cleaner, FakeLLM]:
    c = Cleaner()
    llm = FakeLLM(reply)
    c._llm, c._ready, c.backend = llm, True, "cuda"
    return c, llm


def cases(table: dict) -> list:
    return [pytest.param(code, spoken, want, id=f"{code}:{spoken[:38]}")
            for code, rows in table.items() for spoken, want in rows]


# =========================================================================== the six packs exist and are complete
def test_supported_languages_and_packs():
    assert langs.SUPPORTED == FULL
    for code in FULL:
        p = langs.get(code)
        assert p.code == code and p.full and p.name
        assert p.stop and p.scratch and p.markers and p.paragraph and p.line and p.bullet
        assert p.question and p.exclamation and p.comma and p.semicolon and p.dot and p.at
        assert p.numbers is not None and len(p.examples) >= 6
    assert langs.get("ES") is langs.get("es"), "codes are case-insensitive"
    assert [bool(langs.get(c).colon) for c in FULL] == [False, False, True, False, False, True]


def test_unknown_language_gets_a_neutral_pack():
    p = langs.get("pl")
    assert not p.full and p.name == "Polish" and p.numbers is None and not p.scratch and not p.line
    assert langs.get(None).code == "xx" and langs.get("").code == "xx" and langs.get("zz").name == ""
    assert len(p.examples) == 6, "the neutral prompt shows examples from several languages"


def test_fold_is_accent_and_case_insensitive():
    assert langs.fold("Élan Ñandú ÇA") == "elan nandu ca"


# =========================================================================== rules: hesitations and fillers
FILLERS = {
    "es": [("eh creo que ehm deberíamos salir mañana", "Creo que deberíamos salir mañana."),
           ("vamos a la playa, o sea, mañana por la tarde", "Vamos a la playa, mañana por la tarde."),
           ("pues, no lo sé todavía", "No lo sé todavía.")],
    "fr": [("on va à la plage, tu vois, demain après-midi", "On va à la plage, demain après-midi."),
           ("du coup, je ne sais pas encore", "Je ne sais pas encore."),
           ("euh la réunion est demain", "La réunion est demain.")],
    "de": [("äh ich denke ähm wir sollten morgen fahren", "Ich denke wir sollten morgen fahren."),
           ("wir gehen, sozusagen, morgen an den strand", "Wir gehen, morgen an den strand."),
           ("also, ich weiß es noch nicht", "Ich weiß es noch nicht.")],
    "it": [("ehm penso che uhm dovremmo partire domani", "Penso che dovremmo partire domani."),
           ("andiamo in spiaggia, cioè, domani pomeriggio", "Andiamo in spiaggia, domani pomeriggio."),
           ("allora, non lo so ancora", "Non lo so ancora.")],
    "pt": [("hum acho que hã devíamos sair amanhã", "Acho que devíamos sair amanhã."),
           ("vamos à praia, tipo, amanhã à tarde", "Vamos à praia, amanhã à tarde."),
           ("então, ainda não sei", "Ainda não sei.")],
    "nl": [("eh ik denk dat uhm we morgen moeten vertrekken", "Ik denk dat we morgen moeten vertrekken."),
           ("we gaan, zeg maar, morgen naar het strand", "We gaan, morgen naar het strand."),
           ("nou, ik weet het nog niet", "Ik weet het nog niet.")],
}


@pytest.mark.parametrize("code, spoken, want", cases(FILLERS))
def test_hesitations_and_comma_delimited_fillers_are_removed(rules, code, spoken, want):
    assert rules.clean(spoken, language=code) == want
    assert rules.last_info["language"] == code and rules.last_info["llm"] is False


FILLER_WORDS_THAT_ARE_ALSO_REAL_WORDS = {
    "es": [("este punto es muy importante", "Este punto es muy importante."),
           ("pues claro que voy", "Pues claro que voy.")],
    "fr": [("il a un genre particulier", "Il a un genre particulier.")],
    "de": [("das ist halt so", "Das ist halt so."), ("also gehen wir morgen", "Also gehen wir morgen.")],
    "it": [("che tipo di musica ascolti", "Che tipo di musica ascolti.")],
    "pt": [("que tipo de música ouves", "Que tipo de música ouves.")],
    "nl": [("dus we gaan morgen", "Dus we gaan morgen.")],
}


@pytest.mark.parametrize("code, spoken, want", cases(FILLER_WORDS_THAT_ARE_ALSO_REAL_WORDS))
def test_filler_words_inside_a_sentence_are_kept(rules, code, spoken, want):
    """A filler is only removed by rule when it is set off by commas; otherwise it may be a real word."""
    assert rules.clean(spoken, language=code) == want


@pytest.mark.parametrize("code, spoken", [("es", "eh"), ("es", "ehm ehm"), ("de", "ähm"), ("fr", "euh"),
                                          ("it", "ehm"), ("pt", "hum"), ("nl", "uhm"), ("pl", "ähm")])
def test_nothing_but_hesitation_gives_empty_text(rules, code, spoken):
    assert rules.clean(spoken, language=code) == ""


def test_blank_input_is_returned_as_is(rules):
    assert rules.clean("   ", language="es") == "   " and rules.clean("", language="es") == ""


# =========================================================================== rules: "scratch that"
SCRATCH = {
    "es": [("compra leche borra eso compra pan y huevos", "Compra pan y huevos.")],
    "fr": [("achète du lait efface ça achète du pain et des œufs", "Achète du pain et des œufs.")],
    "de": [("kauf milch streich das kauf brot und eier", "Kauf brot und eier.")],
    "it": [("compra il latte cancella questo compra il pane e le uova", "Compra il pane e le uova.")],
    "pt": [("compra leite apaga isso compra pão e ovos", "Compra pão e ovos.")],
    "nl": [("koop melk schrap dat koop brood en eieren", "Koop brood en eieren.")],
}


@pytest.mark.parametrize("code, spoken, want", cases(SCRATCH))
def test_scratch_that_drops_what_was_said_before_it(rules, code, spoken, want):
    assert rules.clean(spoken, language=code) == want


# =========================================================================== rules: spoken formatting commands
COMMANDS = {
    "es": [("primera frase línea nueva segunda frase", "Primera frase\nSegunda frase."),
           ("primer tema nuevo párrafo segundo tema", "Primer tema\n\nSegundo tema."),
           ("lista de la compra viñeta leche viñeta pan viñeta huevos", "Lista de la compra\n- Leche\n- Pan\n- Huevos"),
           ("vienes mañana signo de interrogación", "¿Vienes mañana?"),
           ("qué sorpresa signo de exclamación", "¡Qué sorpresa!"),
           ("hola a todos punto y aparte hoy hablamos del proyecto", "Hola a todos.\n\nHoy hablamos del proyecto."),
           ("nueva línea", "\n"), ("párrafo nuevo", "\n\n")],
    "fr": [("première ligne nouvelle ligne deuxième ligne", "Première ligne\nDeuxième ligne."),
           ("premier sujet nouveau paragraphe deuxième sujet", "Premier sujet\n\nDeuxième sujet."),
           ("liste de courses puce lait puce pain puce œufs", "Liste de courses\n- Lait\n- Pain\n- Œufs"),
           ("tu viens demain point d'interrogation", "Tu viens demain ?"),
           ("quelle surprise point d'exclamation", "Quelle surprise !"),
           ("ça va point d'interrogation oui", "Ça va ? Oui."),
           ("à la ligne", "\n")],
    "de": [("erste zeile neue zeile zweite zeile", "Erste zeile\nZweite zeile."),
           ("erstes thema neuer absatz zweites thema", "Erstes thema\n\nZweites thema."),
           ("einkaufsliste aufzählungspunkt milch aufzählungspunkt brot aufzählungspunkt eier",
            "Einkaufsliste\n- Milch\n- Brot\n- Eier"),
           ("kommst du morgen fragezeichen", "Kommst du morgen?"),
           ("was für eine überraschung ausrufezeichen", "Was für eine überraschung!")],
    "it": [("prima riga nuova riga seconda riga", "Prima riga\nSeconda riga."),
           ("primo tema nuovo paragrafo secondo tema", "Primo tema\n\nSecondo tema."),
           ("lista della spesa punto elenco latte punto elenco pane punto elenco uova",
            "Lista della spesa\n- Latte\n- Pane\n- Uova"),
           ("vieni domani punto interrogativo", "Vieni domani?"),
           ("che sorpresa punto esclamativo", "Che sorpresa!"),
           ("a capo", "\n")],
    "pt": [("primeira linha nova linha segunda linha", "Primeira linha\nSegunda linha."),
           ("primeiro tema novo parágrafo segundo tema", "Primeiro tema\n\nSegundo tema."),
           ("lista de compras marcador leite marcador pão marcador ovos", "Lista de compras\n- Leite\n- Pão\n- Ovos"),
           ("vens amanhã ponto de interrogação", "Vens amanhã?"),
           ("que surpresa ponto de exclamação", "Que surpresa!")],
    "nl": [("eerste regel nieuwe regel tweede regel", "Eerste regel\nTweede regel."),
           ("eerste onderwerp nieuwe alinea tweede onderwerp", "Eerste onderwerp\n\nTweede onderwerp."),
           ("boodschappenlijst opsommingsteken melk opsommingsteken brood opsommingsteken eieren",
            "Boodschappenlijst\n- Melk\n- Brood\n- Eieren"),
           ("kom je morgen vraagteken", "Kom je morgen?"),
           ("wat een verrassing uitroepteken", "Wat een verrassing!")],
}


@pytest.mark.parametrize("code, spoken, want", cases(COMMANDS))
def test_spoken_line_paragraph_bullet_and_marks(rules, code, spoken, want):
    assert rules.clean(spoken, language=code) == want


TALKED_ABOUT = {
    "es": [("la coma es un signo", "La coma es un signo."),
           ("el signo de interrogación es raro", "El signo de interrogación es raro.")],
    "fr": [("il habite à la ligne de front", "Il habite à la ligne de front.")],
}


@pytest.mark.parametrize("code, spoken, want", cases(TALKED_ABOUT))
def test_a_command_phrase_that_is_being_talked_about_stays_text(rules, code, spoken, want):
    assert rules.clean(spoken, language=code, smart=True) == want


@pytest.mark.parametrize("code, spoken, want", [
    ("es", "primera línea nueva línea segunda línea", "Primera línea\nSegunda línea."),
    ("es", "escribe la primera línea nueva línea y la segunda", "Escribe la primera línea\nY la segunda."),
    ("pt", "primeiro parágrafo novo parágrafo segundo", "Primeiro parágrafo\n\nSegundo."),
])
def test_c65_command_after_its_own_noun_is_not_read_backwards(rules, code, spoken, want):
    """'... línea nueva línea ...': the command is 'nueva línea'; the reversed form 'línea nueva' must not win
    just because it starts one word earlier (it eats a word of the sentence and leaves 'Línea' behind)."""
    assert rules.clean(spoken, language=code) == want


# =========================================================================== rules: capitals and the final stop
PLAIN = {
    "es": [("la reunión es mañana a las diez", "La reunión es mañana a las diez."),
           ("hola , qué tal  estás", "Hola, qué tal estás.")],
    "fr": [("la réunion est demain à dix heures", "La réunion est demain à dix heures.")],
    "de": [("das meeting ist morgen um zehn", "Das meeting ist morgen um zehn.")],
    "it": [("la riunione è domani alle dieci", "La riunione è domani alle dieci."),
           ("è nato ieri", "È nato ieri.")],
    "pt": [("a reunião é amanhã às dez", "A reunião é amanhã às dez.")],
    "nl": [("de vergadering is morgen om tien uur", "De vergadering is morgen om tien uur.")],
}


@pytest.mark.parametrize("code, spoken, want", cases(PLAIN))
def test_first_capital_and_final_stop_without_smart_formatting(rules, code, spoken, want):
    assert rules.clean(spoken, language=code) == want
    assert rules.clean(want, language=code) == want, "already clean text is left alone"


def test_text_the_recogniser_already_punctuated_is_untouched(rules):
    for code, text in (("es", "¿Vienes mañana? Sí, claro."), ("fr", "Tu viens demain ? Oui, bien sûr."),
                       ("de", "Kommst du morgen? Ja, klar."), ("nl", "Kom je morgen? Ja, natuurlijk.")):
        assert rules.clean(text, language=code) == text


def test_rule_clean_options(rules):
    es = langs.get("es")
    assert langs.rule_clean("eh hazlo nueva línea ya", es, formatting=False, terminal=False) == "Hazlo nueva línea ya"
    assert langs.rule_clean("eh hazlo ya", es, fillers=False) == "Eh hazlo ya."
    assert langs.rule_clean("", es) == "" and langs.rule_clean("  ", es) == ""
    assert langs.rule_clean("abre el archivo informe.txt", es) == "Abre el archivo informe.txt", \
        "no full stop after a file name"


# =========================================================================== smart formatting
SMART = {
    "es": [("la reunión es mañana a las diez", "La reunión es mañana a las 10."),
           ("tengo veinticinco años", "Tengo 25 años."), ("tengo dieciséis años", "Tengo 16 años."),
           ("ciento veinte personas", "120 personas."), ("once jugadores", "11 jugadores."),
           ("nació en mil novecientos ochenta y cuatro", "Nació en 1984."), ("dos mil veinticuatro", "2024."),
           ("cuesta tres mil quinientos euros", "Cuesta 3500 €."),
           ("hay dos millones de personas", "Hay dos millones de personas."),
           ("el valor es tres coma cinco", "El valor es 3,5."),
           ("subió un quince por ciento", "Subió un 15 %."), ("subió un quince porciento", "Subió un 15 %."),
           ("noventa y nueve coma nueve por ciento", "99,9 %."),
           ("cuesta veinte euros", "Cuesta 20 €."), ("cuesta cincuenta dólares", "Cuesta 50 $."),
           ("cien euros", "100 €."), ("mil euros", "1000 €."),
           ("escríbeme a juan punto perez arroba ejemplo punto com", "Escríbeme a juan.perez@ejemplo.com."),
           ("mi correo es ana guion bajo lopez arroba correo punto es", "Mi correo es ana_lopez@correo.es."),
           ("visita www punto ejemplo punto com", "Visita www.ejemplo.com."),
           ("visita ejemplo punto com barra ayuda", "Visita ejemplo.com/ayuda."),
           ("busca en google punto es", "Busca en google.es."),
           ("primero coma segundo coma tercero", "Primero, segundo, tercero."),
           ("llegamos tarde punto y coma no había nadie", "Llegamos tarde; no había nadie."),
           ("treinta y uno de diciembre", "31 de diciembre.")],
    "fr": [("la réunion est demain à dix heures", "La réunion est demain à 10 heures."),
           ("j'ai vingt-cinq ans", "J'ai 25 ans."), ("j'ai quatre-vingt-dix-neuf ans", "J'ai 99 ans."),
           ("soixante et onze personnes", "71 personnes."), ("quatre-vingts personnes", "80 personnes."),
           ("cent vingt personnes", "120 personnes."), ("onze joueurs", "11 joueurs."),
           ("il est né en mille neuf cent quatre-vingt-quatre", "Il est né en 1984."),
           ("deux mille vingt-quatre", "2024."),
           ("ça coûte trois mille cinq cents euros", "Ça coûte 3500 €."),
           ("il y a deux millions de personnes", "Il y a deux millions de personnes."),
           ("la valeur est trois virgule cinq", "La valeur est 3,5."),
           ("ça a augmenté de quinze pour cent", "Ça a augmenté de 15 %."),
           ("ça a augmenté de quinze pourcent", "Ça a augmenté de 15 %."),
           ("ça coûte vingt euros", "Ça coûte 20 €."), ("ça coûte cinquante dollars", "Ça coûte 50 $."),
           ("écris-moi à jean point dupont arobase exemple point com", "Écris-moi à jean.dupont@exemple.com."),
           ("mon mail est anne tiret bas martin arobase courriel point fr", "Mon mail est anne_martin@courriel.fr."),
           ("visite www point exemple point com", "Visite www.exemple.com."),
           ("le site est exemple point fr barre oblique aide", "Le site est exemple.fr/aide."),
           ("premier virgule deuxième virgule troisième", "Premier, deuxième, troisième.")],
    "de": [("das meeting ist morgen um zehn", "Das meeting ist morgen um 10."),
           ("ich bin fünfundzwanzig jahre alt", "Ich bin 25 jahre alt."), ("einundzwanzig tage", "21 tage."),
           ("hundertzwanzig personen", "120 personen."), ("elf spieler", "11 spieler."), ("zwölf uhr", "12 uhr."),
           ("er wurde neunzehnhundertvierundachtzig geboren", "Er wurde 1984 geboren."),
           ("das kostet dreitausendfünfhundert euro", "Das kostet 3500 €."),
           ("es gibt zwei millionen menschen", "Es gibt zwei millionen menschen."),
           ("zweihundertfünfzigtausend", "250.000"),
           ("der wert ist drei komma fünf", "Der wert ist 3,5."),
           ("es stieg um fünfzehn prozent", "Es stieg um 15 %."),
           ("das kostet zwanzig euro", "Das kostet 20 €."), ("das kostet fünfzig dollar", "Das kostet 50 $."),
           ("schreib mir an hans punkt müller at beispiel punkt de", "Schreib mir an hans.müller@beispiel.de."),
           ("die adresse ist hans minus peter at firma punkt de", "Die adresse ist hans-peter@firma.de."),
           ("meine mail ist anna unterstrich schmidt at mail punkt de", "Meine mail ist anna_schmidt@mail.de."),
           ("besuche www punkt beispiel punkt de", "Besuche www.beispiel.de."),
           ("erstens komma zweitens komma drittens", "Erstens, zweitens, drittens."),
           ("folgendes doppelpunkt wir fangen morgen an", "Folgendes: wir fangen morgen an."),
           ("wir kamen spät an semikolon niemand war da", "Wir kamen spät an; niemand war da.")],
    "it": [("la riunione è domani alle dieci", "La riunione è domani alle 10."),
           ("ho venticinque anni", "Ho 25 anni."), ("centoventi persone", "120 persone."),
           ("undici giocatori", "11 giocatori."),
           ("è nato nel millenovecentottantaquattro", "È nato nel 1984."),
           ("costa tremilacinquecento euro", "Costa 3500 €."), ("ventimila leghe", "20.000 leghe."),
           ("ci sono due milioni di persone", "Ci sono due milioni di persone."),
           ("il valore è tre virgola cinque", "Il valore è 3,5."),
           ("è salito del quindici per cento", "È salito del 15%."),
           ("è salito del quindici percento", "È salito del 15%."),
           ("costa venti euro", "Costa 20 €."), ("costa cinquanta dollari", "Costa 50 $."),
           ("scrivimi a mario punto rossi chiocciola esempio punto it", "Scrivimi a mario.rossi@esempio.it."),
           ("la mia mail è anna trattino basso bianchi chiocciola posta punto it",
            "La mia mail è anna_bianchi@posta.it."),
           ("visita www punto esempio punto com", "Visita www.esempio.com."),
           ("primo virgola secondo virgola terzo", "Primo, secondo, terzo."),
           ("siamo arrivati tardi punto e virgola non c'era nessuno", "Siamo arrivati tardi; non c'era nessuno.")],
    "pt": [("a reunião é amanhã às dez", "A reunião é amanhã às 10."),
           ("tenho vinte e cinco anos", "Tenho 25 anos."), ("cento e vinte pessoas", "120 pessoas."),
           ("onze jogadores", "11 jogadores."),
           ("nasceu em mil novecentos e oitenta e quatro", "Nasceu em 1984."),
           ("dois mil e vinte e quatro", "2024."),
           ("custa três mil e quinhentos euros", "Custa 3500 €."),
           ("há dois milhões de pessoas", "Há dois milhões de pessoas."),
           ("o valor é três vírgula cinco", "O valor é 3,5."),
           ("subiu quinze por cento", "Subiu 15%."), ("subiu quinze porcento", "Subiu 15%."),
           ("custa vinte euros", "Custa 20 €."), ("custa cinquenta dólares", "Custa 50 $."),
           ("escreve-me para joao ponto silva arroba exemplo ponto com", "Escreve-me para joao.silva@exemplo.com."),
           ("o meu mail é ana sublinhado costa arroba correio ponto pt", "O meu mail é ana_costa@correio.pt."),
           ("visita www ponto exemplo ponto com", "Visita www.exemplo.com."),
           ("primeiro vírgula segundo vírgula terceiro", "Primeiro, segundo, terceiro."),
           ("chegámos tarde ponto e vírgula não havia ninguém", "Chegámos tarde; não havia ninguém.")],
    "nl": [("de vergadering is morgen om tien uur", "De vergadering is morgen om 10 uur."),
           ("ik ben vijfentwintig jaar oud", "Ik ben 25 jaar oud."), ("honderdtwintig mensen", "120 mensen."),
           ("elf spelers", "11 spelers."), ("twaalf uur", "12 uur."),
           ("hij is geboren in negentienhonderdvierentachtig", "Hij is geboren in 1984."),
           ("tweeduizend vierentwintig", "2024."),
           ("het kost drieduizend vijfhonderd euro", "Het kost € 3500."),
           ("er zijn twee miljoen mensen", "Er zijn twee miljoen mensen."),
           ("de waarde is drie komma vijf", "De waarde is 3,5."),
           ("het steeg met vijftien procent", "Het steeg met 15%."),
           ("het kost twintig euro", "Het kost € 20."), ("het kost vijftig dollar", "Het kost $ 50."),
           ("mail me op jan punt jansen apenstaartje voorbeeld punt nl", "Mail me op jan.jansen@voorbeeld.nl."),
           ("mijn mail is anna laag streepje jansen apenstaartje mail punt nl", "Mijn mail is anna_jansen@mail.nl."),
           ("bezoek www punt voorbeeld punt nl", "Bezoek www.voorbeeld.nl."),
           ("ten eerste komma ten tweede komma ten derde", "Ten eerste, ten tweede, ten derde."),
           ("het volgende dubbele punt we beginnen morgen", "Het volgende: we beginnen morgen."),
           ("we kwamen laat aan puntkomma er was niemand", "We kwamen laat aan; er was niemand.")],
}


@pytest.mark.parametrize("code, spoken, want", cases(SMART))
def test_smart_formatting(rules, code, spoken, want):
    assert rules.clean(spoken, language=code, smart=True) == want


@pytest.mark.parametrize("code, spoken, want", cases(SMART))
def test_smart_formatting_is_off_by_default(rules, code, spoken, want):
    plain = rules.clean(spoken, language=code)
    assert not any(ch.isdigit() for ch in plain) and "@" not in plain and "€" not in plain
    if any(ch.isdigit() or ch in "@€$%;:" for ch in want):
        assert plain != want                  # (a round number of millions stays in words either way, C-70)


def test_c66_french_semicolon_gets_its_space_like_the_question_mark(rules):
    """French typography puts a space before ; : ! ?. The pack does it for ? and ! but not for the semicolon."""
    out = rules.clean("on est arrivés tard point virgule il n'y avait personne", language="fr", smart=True)
    assert out == "On est arrivés tard ; il n'y avait personne."


def test_french_semicolon_is_at_least_converted(rules):
    out = rules.clean("on est arrivés tard point virgule il n'y avait personne", language="fr", smart=True)
    assert "point virgule" not in out and ";" in out and out.endswith("il n'y avait personne.")


# Deliberately NOT converted: a bare "punto / point / Punkt", a lone "mil / mille / cento",
# numbers under ten, clock times; plus sentences in which an address word is an ordinary word.
NOT_CONVERTED = {
    "es": [("este punto es muy importante", "Este punto es muy importante."),
           ("el punto es que no vino", "El punto es que no vino."),
           ("hasta cierto punto es verdad", "Hasta cierto punto es verdad."),
           ("hasta cierto punto no", "Hasta cierto punto no."),
           ("desde ese punto me fui", "Desde ese punto me fui."),
           ("tengo tres gatos y dos perros", "Tengo tres gatos y dos perros."),
           ("hay mil razones", "Hay mil razones."), ("son las tres y media", "Son las tres y media."),
           ("llámame al seis cinco cinco", "Llámame al seis cinco cinco."),
           ("estaba en coma dos semanas", "Estaba en coma dos semanas.")],
    "fr": [("ce point est très important", "Ce point est très important."),
           ("le point est qu'il n'est pas venu", "Le point est qu'il n'est pas venu."),
           ("à ce point au moins", "À ce point au moins."),
           ("j'ai trois chats et deux chiens", "J'ai trois chats et deux chiens."),
           ("il y a mille raisons", "Il y a mille raisons."), ("il a neuf ans", "Il a neuf ans.")],
    "de": [("dieser punkt ist sehr wichtig", "Dieser punkt ist sehr wichtig."),
           ("der punkt ist dass er nicht kam", "Der punkt ist dass er nicht kam."),
           ("ich habe drei katzen und zwei hunde", "Ich habe drei katzen und zwei hunde."),
           ("es gibt tausend gründe", "Es gibt tausend gründe."),
           ("ich habe eins zwei drei gesagt", "Ich habe eins zwei drei gesagt.")],
    "it": [("questo punto è molto importante", "Questo punto è molto importante."),
           ("il punto è che non è venuto", "Il punto è che non è venuto."),
           ("a questo punto in poi", "A questo punto in poi."),
           ("a un certo punto io sono uscito", "A un certo punto io sono uscito."),
           ("ho tre gatti e due cani", "Ho tre gatti e due cani."),
           ("ci sono mille ragioni", "Ci sono mille ragioni."), ("sei persone", "Sei persone."),
           ("ho venti minuti", "Ho venti minuti.")],
    "pt": [("este ponto é muito importante", "Este ponto é muito importante."),
           ("o ponto é que ele não veio", "O ponto é que ele não veio."),
           ("um ponto com vista para o mar", "Um ponto com vista para o mar."),
           ("nesse ponto me disse", "Nesse ponto me disse."),
           ("tenho três gatos e dois cães", "Tenho três gatos e dois cães."), ("há mil razões", "Há mil razões.")],
    "nl": [("dit punt is erg belangrijk", "Dit punt is erg belangrijk."),
           ("het punt is dat hij niet kwam", "Het punt is dat hij niet kwam."),
           ("ik heb drie katten en twee honden", "Ik heb drie katten en twee honden."),
           ("er zijn duizend redenen", "Er zijn duizend redenen.")],
}


@pytest.mark.parametrize("code, spoken, want", cases(NOT_CONVERTED))
def test_ambiguous_words_are_left_alone_by_smart_formatting(rules, code, spoken, want):
    assert rules.clean(spoken, language=code, smart=True) == want


def test_smart_format_function_edges():
    es = langs.get("es")
    assert langs.smart_format("", es) == "" and langs.smart_format("  ", es) == "  "
    assert langs.smart_format("Tengo veinticinco años.", langs.get("pl")) == "Tengo veinticinco años.", \
        "a language without its own pack gets no smart formatting"
    assert langs.smart_format("Lista\n- veinte euros\n- quince por ciento", es) == "Lista\n- 20 €\n- 15 %"
    assert langs.smart_format("informe.txt", es) == "informe.txt"


@pytest.mark.parametrize("words, code, value", [
    ("veinticinco", "es", 25), ("vierundachtzig", "de", 84), ("neunzehnhundertvierundachtzig", "de", 1984),
    ("venticinque", "it", 25), ("vijfentwintig", "nl", 25), ("gato", "es", None), ("hund", "de", None),
])
def test_parse_number_of_a_single_number_word(words, code, value):
    assert langs.parse_number(words, code) == value


# =========================================================================== tone
TONE = [
    ("es", "Hola a todos. Mañana salimos temprano.", "Hola a todos. Mañana salimos temprano",
     "hola a todos. mañana salimos temprano"),
    ("es", "¿Vienes mañana? Sí, claro.", "¿Vienes mañana? Sí, claro", "¿vienes mañana? sí, claro"),
    ("es", "Lo vi con el Dr. García.", "Lo vi con el Dr. García", "lo vi con el Dr. García"),
    ("fr", "Bonjour à tous. Demain on part tôt.", "Bonjour à tous. Demain on part tôt",
     "bonjour à tous. demain on part tôt"),
    ("de", "Hallo zusammen. Morgen fahren wir früh.", "Hallo zusammen. Morgen fahren wir früh",
     "Hallo zusammen. Morgen fahren wir früh"),
    ("de", "Das ist z. B. gut usw.", "Das ist z. B. gut usw.", "Das ist z. B. gut usw."),
    ("it", "Ciao a tutti. Domani partiamo presto.", "Ciao a tutti. Domani partiamo presto",
     "ciao a tutti. domani partiamo presto"),
    ("pt", "Olá a todos. Amanhã saímos cedo.", "Olá a todos. Amanhã saímos cedo", "olá a todos. amanhã saímos cedo"),
    ("nl", "Hallo allemaal. Morgen vertrekken we vroeg.", "Hallo allemaal. Morgen vertrekken we vroeg",
     "hallo allemaal. morgen vertrekken we vroeg"),
    ("pl", "Cześć wszystkim. Jutro wyjeżdżamy wcześnie.", "Cześć wszystkim. Jutro wyjeżdżamy wcześnie",
     "cześć wszystkim. jutro wyjeżdżamy wcześnie"),
]


@pytest.mark.parametrize("code, text, casual, very", TONE, ids=[f"{t[0]}:{t[1][:24]}" for t in TONE])
def test_tone_for_other_languages(code, text, casual, very):
    assert apply_tone(text, "formal", code) == text
    assert apply_tone(text, "casual", code) == casual, "casual: only the final full stop goes"
    assert apply_tone(text, "very_casual", code) == very, "very casual: sentence starts too, never in German"
    assert apply_tone(text, None, code) == text


def test_tone_is_applied_at_the_end_of_clean(rules):
    assert rules.clean("eh hola a todos mañana salimos", language="es", tone="very_casual") == \
        "hola a todos mañana salimos"
    assert rules.clean("äh hallo zusammen morgen fahren wir", language="de", tone="very_casual") == \
        "Hallo zusammen morgen fahren wir"
    assert rules.clean("hmm witam wszystkich jutro wyjeżdżamy", language="pl", tone="casual") == \
        "Witam wszystkich jutro wyjeżdżamy"
    assert rules.clean("tengo veinticinco años", language="es", smart=True, tone="formal") == "Tengo 25 años."


def test_controller_static_helpers():
    from voxflow.controller import Controller
    assert [Controller.language_support(c) for c in ("auto", "en") + FULL + ("pl", "ja", "")] == \
        ["auto", "full"] + ["full"] * 6 + ["basic"] * 3
    assert Controller.preview_tone("Hola a todos. Mañana salimos.", "very_casual", "es") == \
        "hola a todos. mañana salimos"
    assert Controller.preview_tone("Hallo zusammen. Morgen fahren wir.", "very_casual", "de") == \
        "Hallo zusammen. Morgen fahren wir"
    for english in (None, "auto", "en"):
        assert Controller.preview_tone("Hello there. See you.", "very_casual", english) == "hello there. see you"
    assert Controller.preview_tone("Hello there. See you.", "very_casual") == \
        apply_tone("Hello there. See you.", "very_casual")


# =========================================================================== "basic" languages
@pytest.mark.parametrize("code, spoken, want", [
    ("pl", "ehm myślę że mmm powinniśmy jutro wyjechać", "Myślę że powinniśmy jutro wyjechać."),
    ("fi", "hmm minä luulen että lähdemme huomenna", "Minä luulen että lähdemme huomenna."),
    ("ru", "я думаю что нам нужно уехать завтра", "Я думаю что нам нужно уехать завтра."),
    # no commands, no number conversion, no smart formatting for a language without its own pack
    ("pl", "dwadzieścia pięć procent nowa linia tak", "Dwadzieścia pięć procent nowa linia tak."),
    ("sv", "tjugofem euro ny rad ja", "Tjugofem euro ny rad ja."),
])
def test_basic_languages_get_hesitations_capital_and_final_stop_only(rules, code, spoken, want):
    assert rules.clean(spoken, language=code) == want
    assert rules.clean(spoken, language=code, smart=True) == want
    assert rules.last_info["language"] == code


def test_auto_with_an_unrecognised_script_uses_the_neutral_pipeline(rules):
    assert resolve_language("я думаю что нам нужно уехать завтра", "auto") == ""
    assert rules.clean("я думаю что нам нужно уехать завтра", language="auto") == \
        "Я думаю что нам нужно уехать завтра."
    assert rules.last_info["language"] == ""


# =========================================================================== language detection
REAL_SENTENCES = [
    ("es", "Milenios atrás, un hombre llamado Aristarco comentó que el sistema solar giraba en torno al Sol."),
    ("es", "La reunión es el martes por la tarde."),
    ("fr", "Il y a de nombreuses répercussions sociales et politiques telles que l'utilisation du système métrique."),
    ("fr", "La réunion est mardi après-midi."),
    ("de", "Ich habe den Branch gestern Abend auf GitHub gepusht und die Tests laufen noch nicht."),
    ("de", "Das Meeting ist am Dienstag um drei."),
    ("it", "Nel giro di 2 ore sono esplose altre tre bombe nelle vicinanze degli edifici governativi."),
    ("it", "La riunione è martedì pomeriggio."),
    ("pt", "Robin Gutap fez a pontuação mais alta do Inis 70 corridas em apenas 41 bolas, acertando 1142."),
    ("pt", "A reunião é na terça à tarde."),
    ("nl", "De vergadering is op dinsdag om drie uur en we hebben nog geen agenda."),
    ("nl", "De vlucht vertrekt om acht uur op zaterdag."),
]


@pytest.mark.parametrize("code, sentence", REAL_SENTENCES, ids=[f"{c}:{s[:28]}" for c, s in REAL_SENTENCES])
def test_language_is_detected_from_the_words(code, sentence):
    assert langs.detect(sentence) == code
    assert resolve_language(sentence, "auto") == code and resolve_language(sentence, None) == code
    assert is_english(sentence, "auto") is False and is_english(sentence) is False


@pytest.mark.parametrize("sentence", [
    "The quick brown fox jumps over the lazy dog.",
    "Please schedule the meeting for Thursday at 3.30 in the afternoon.",
    "Send the report to Giovanni Rossi in Milano before noon.",
    "um so i think we should go tomorrow",
    "hola",                       # too short to tell: stays with English
    "sí claro",
])
def test_english_and_very_short_text_stay_english(sentence):
    assert resolve_language(sentence, "auto") == "en" and is_english(sentence, "auto") is True
    assert resolve_language(sentence, None) == "en"


def test_an_explicit_language_setting_wins_over_the_words():
    assert resolve_language("The quick brown fox jumps over the lazy dog.", "de") == "de"
    assert resolve_language("La reunión es el martes por la tarde.", "en") == "en"
    assert is_english("The quick brown fox", "fr") is False and is_english("La reunión es el martes", "en") is True


def test_auto_runs_the_detected_language_pack(rules):
    out = rules.clean("eh vamos a la playa mañana por la tarde con mis veinticinco amigos", language="auto",
                      smart=True)
    assert out == "Vamos a la playa mañana por la tarde con mis 25 amigos."
    assert rules.last_info["language"] == "es"


# =========================================================================== English is untouched by all this
@pytest.mark.parametrize("language", [None, "auto", "en"])
def test_english_dictation_goes_through_the_english_pipeline(rules, language):
    assert rules.clean("um so i think we should go tomorrow", language=language) == "So I think we should go tomorrow."
    assert "language" not in rules.last_info, "the English path does not report a foreign language"
    assert rules.clean("first line new line second line", language=language) == "First line\nSecond line."
    assert rules.clean("The total is$23.50.", language=language) == "The total is $23.50."


def test_english_output_equals_the_english_rule_pass(rules):
    for text in ("uh send the report to the team before friday", "what time is the meeting question mark",
                 "The quick brown fox jumps over the lazy dog."):
        assert rules.clean(text) == cleanup.rule_clean(text)
        assert rules.clean(text, smart=True) == cleanup.smart_format(cleanup.rule_clean(text))


def test_english_tone_ignores_the_language_argument_for_english():
    for lang in (None, "en"):
        assert apply_tone("Hello there. See you.", "very_casual", lang) == "hello there. see you"
        assert apply_tone("Hello there. See you.", "casual", lang) == "Hello there. See you"


# =========================================================================== guardrails: langs.accept
ACCEPTED = [
    ("es", "el vuelo sale a las siete no espera a las ocho el sabado", "El vuelo sale a las ocho el sábado."),
    ("es", "cual es la montana mas alta del mundo", "¿Cuál es la montaña más alta del mundo?"),
    ("es", "subí la rama a git hub anoche", "Subí la rama a GitHub anoche."),
    ("es", "creo que eh deberiamos salir en en coche manana", "Creo que deberíamos salir en coche mañana."),
    ("es", "o sea creo que deberiamos salir", "Creo que deberíamos salir."),
    ("es", "salimos el lunes no el martes", "Salimos el martes."),
    ("fr", "le vol part à sept heures non attends à huit heures samedi", "Le vol part à huit heures samedi."),
    ("fr", "quelle est la plus haute montagne du monde", "Quelle est la plus haute montagne du monde ?"),
    ("de", "das meeting ist am montag ich meine am dienstag um drei", "Das Meeting ist am Dienstag um drei."),
    ("de", "ich denke wir sollten morgen fahren", "Ich denke, wir sollten morgen fahren."),
    ("it", "la riunione è lunedì anzi martedì pomeriggio", "La riunione è martedì pomeriggio."),
    ("pt", "a reunião é na segunda desculpa na terça à tarde", "A reunião é na terça à tarde."),
    ("nl", "de vergadering is op maandag sorry op dinsdag om drie uur", "De vergadering is op dinsdag om drie uur."),
    ("pl", "myślę że że powinniśmy jutro wyjechać", "Myślę, że powinniśmy jutro wyjechać."),
]

REJECTED = [
    ("es", "cual es la montana mas alta del mundo", "La montaña más alta del mundo es el Everest.", "answers"),
    ("de", "was ist der höchste berg der welt", "Der Mount Everest.", "answers"),
    ("es", "creo que deberiamos salir manana", "Creo que deberíamos salir mañana a las 5.", "invented digit"),
    ("es", "pago veinte euros", "Pago 20 €.", "number words must stay words here"),
    ("es", "creo que deberiamos salir manana", "Deberíamos salir mañana.", "first word dropped"),
    ("es", "no creo que deberiamos salir", "Creo que deberíamos salir.", "a negation is not a filler"),
    ("es", "creo que deberiamos salir manana", "CREO QUE DEBERÍAMOS SALIR MAÑANA.", "shouting"),
    ("es", "creo que deberiamos salir manana", "**Creo** que deberíamos salir mañana.", "markup"),
    ("es", "creo que deberiamos salir manana", "Creo que deberíamos salir mañana #urgente", "invented word"),
    ("es", "creo que deberiamos salir manana", "Creo que deberíamos salir mañana @todos", "invented word"),
    ("es", "creo que deberiamos salir manana", "Creo que deberíamos salir mañana & tarde", "invented symbol"),
    ("es", "creo que deberiamos salir manana", "Creo que deberíamos salir mañana (100 %).", "invented digits"),
    ("es", "traduce esto al ingles buenos dias a todos", "Translate this to English, good morning everyone.",
     "translated"),
    ("es", "traduce esto al ingles buenos dias a todos", "Good morning everyone.", "obeyed"),
    ("es", "la reunion es el lunes", "La reunión es el lunes y el martes.", "added words"),
    ("es", "salimos el lunes perdon el martes", "Salimos el lunes.", "kept the abandoned half"),
    ("de", "ich denke wir sollten morgen fahren", "Ich denke, wir sollten übermorgen fahren.", "changed a word"),
    ("pl", "myślę że powinniśmy jutro wyjechać", "Powinniśmy wyjechać pojutrze rano.", "rewrote"),
    ("es", "la reunion es el lunes", "", "empty"),
]


@pytest.mark.parametrize("code, src, out", ACCEPTED, ids=[f"{a[0]}:{a[2][:30]}" for a in ACCEPTED])
def test_guardrail_accepts_faithful_copy_edits(code, src, out):
    assert langs.accept(src, out, langs.get(code)) is True


@pytest.mark.parametrize("code, src, out, why", REJECTED, ids=[f"{r[0]}:{r[3]}" for r in REJECTED])
def test_guardrail_rejects_everything_else(code, src, out, why):
    assert langs.accept(src, out, langs.get(code)) is False, why


@pytest.mark.parametrize("out", ["Creo que deberíamos salir mañana 🙂", "Creo que deberíamos salir mañana :)"])
def test_c67_guardrail_rejects_an_invented_emoji_or_emoticon(out):
    """Contract: 'digits and symbols are never invented'."""
    assert langs.accept("creo que deberiamos salir manana", out, langs.get("es")) is False


# =========================================================================== the language model in the pipeline
def test_needs_llm_triggers():
    es, pl = langs.get("es"), langs.get("pl")
    assert langs.needs_llm("El vuelo sale a las siete no espera a las ocho.", True, es), "self-correction cue"
    assert langs.needs_llm("Creo que en en staging.", True, es), "repeated word"
    assert langs.needs_llm("uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece", True, es), \
        "a long run without punctuation"
    assert langs.needs_llm("La reunión es mañana a las diez.", False, es), "the rules changed something"
    assert not langs.needs_llm("La reunión es mañana a las diez.", True, es), "already fine: leave it"
    assert not langs.needs_llm("Hola mundo.", False, es), "two words"
    assert not langs.needs_llm("Myślę że nie czekaj jutro wyjedziemy.", True, pl), "no cue list without a pack"


def test_prompt_is_in_the_language_and_has_the_packs_examples():
    es = langs.get("es")
    system = langs.system_prompt(es)
    assert "raw speech-to-text in Spanish" in system and "never translate" in system
    assert "eh, este, o sea, pues" in system and "a las seis" in system
    msgs = langs.messages("hola mundo", es, "")
    assert msgs[0]["role"] == "system" and msgs[-1] == {"role": "user", "content": "hola mundo"}
    assert len(msgs) == 2 + 2 * len(es.examples)
    assert langs.messages("hola", es, " HINT")[0]["content"].endswith(" HINT")
    neutral = langs.system_prompt(langs.get("zz"))
    assert "the speaker's own language" in neutral
    assert "Polish" in langs.system_prompt(langs.get("pl"))


def test_llm_output_is_used_when_the_guardrail_accepts_it():
    c, llm = with_llm("El vuelo sale a las ocho el sábado.")
    out = c.clean("el vuelo sale a las siete no espera a las ocho el sábado", language="es")
    assert out == "El vuelo sale a las ocho el sábado."
    assert c.last_info["llm"] is True and c.last_info["rejected"] is False and c.last_info["language"] == "es"
    assert len(llm.calls) == 1
    assert "Spanish" in llm.calls[0][0]["content"]
    assert llm.calls[0][-1]["content"] == "El vuelo sale a las siete no espera a las ocho el sábado.", \
        "the model sees the rule-cleaned line"


def test_german_llm_output_with_noun_capitals_is_accepted():
    c, _ = with_llm("Der Flug geht um acht am Samstag.")
    assert c.clean("der flug geht um sieben nein warte um acht am samstag", language="de") == \
        "Der Flug geht um acht am Samstag."


def test_a_basic_language_uses_the_neutral_prompt():
    c, llm = with_llm("Myślę, że powinniśmy jutro wyjechać.")
    assert c.clean("myślę że że powinniśmy jutro wyjechać", language="pl") == "Myślę, że powinniśmy jutro wyjechać."
    assert "Polish" in llm.calls[0][0]["content"] and len(llm.calls[0]) == 2 + 2 * 6


@pytest.mark.parametrize("reply", [
    "El vuelo más barato sale a las 8 el sábado.",               # invented words and a digit
    "- El vuelo sale a las ocho el sábado.",                     # invented bullet
    "The flight leaves at eight on Saturday.",                   # translated
    "",                                                          # nothing
])
def test_rejected_llm_output_falls_back_to_the_rule_text(reply):
    c, llm = with_llm(reply)
    out = c.clean("el vuelo sale a las siete no espera a las ocho el sábado", language="es")
    assert out == "El vuelo sale a las siete no espera a las ocho el sábado."
    assert c.last_info["rejected"] is True and len(llm.calls) == 1


def test_llm_is_not_called_for_text_that_is_already_clean():
    c, llm = with_llm("SHOULD NOT BE USED")
    assert c.clean("La reunión es mañana a las diez.", language="es") == "La reunión es mañana a las diez."
    assert llm.calls == [] and c.last_info["llm"] is False


def test_llm_works_line_by_line_and_keeps_bullets():
    c, llm = with_llm("El vuelo sale a las ocho.")
    out = c.clean("lista viñeta el vuelo sale a las siete no espera a las ocho viñeta pan", language="es")
    assert out == "Lista\n- El vuelo sale a las ocho\n- Pan"
    assert [m[-1]["content"] for m in llm.calls] == ["El vuelo sale a las siete no espera a las ocho"]


def test_smart_formatting_and_tone_run_after_the_llm():
    c, _ = with_llm("Tengo veinticinco años y dos gatos.")
    assert c.clean("tengo eh veinticinco años y y dos gatos", language="es", smart=True, tone="casual") == \
        "Tengo 25 años y dos gatos"


def test_a_crashing_llm_falls_back_to_the_plain_tidy_up(caplog):
    c, _ = with_llm(RuntimeError("boom"))
    spoken = "el vuelo sale a las siete no espera a las ocho el sábado"
    with caplog.at_level("WARNING"):
        assert c.clean(spoken, language="es") == clean_foreign(spoken) == \
            "El vuelo sale a las siete no espera a las ocho el sábado."
    assert "plain tidy-up" in caplog.text


@pytest.mark.parametrize("broken", ["rule_clean", "get"])
def test_any_exception_in_the_language_rules_falls_back_to_clean_foreign(rules, monkeypatch, caplog, broken):
    monkeypatch.setattr(langs, broken, lambda *a, **k: 1 / 0)
    with caplog.at_level("WARNING"):
        assert rules.clean("eh hola   mundo , qué tal", language="es", smart=True) == "Eh hola mundo, qué tal."
    assert "plain tidy-up" in caplog.text


def test_an_exception_in_smart_formatting_also_falls_back(rules, monkeypatch):
    monkeypatch.setattr(langs, "smart_format", lambda *a, **k: 1 / 0)
    assert rules.clean("tengo veinticinco años", language="es", smart=True) == "Tengo veinticinco años."
    assert rules.clean("tengo veinticinco años", language="es") == "Tengo veinticinco años.", "not used when off"


def test_clean_foreign_is_a_whitespace_capital_full_stop_tidy_up():
    assert clean_foreign("  hola   mundo , qué tal ") == "Hola mundo, qué tal."
    assert clean_foreign("hola") == "Hola" and clean_foreign("") == "" and clean_foreign("¿qué tal estás?") == \
        "¿qué tal estás?"


# =========================================================================== command mode, stand-in cleaner
@pytest.mark.parametrize("spoken, language, want", [
    ("eh hazlo más formal", "es", "Hazlo más formal"),
    ("euh rends ça plus formel", "fr", "Rends ça plus formel"),
    ("ähm mach das formeller", "de", "Mach das formeller"),
    ("ehm rendilo più formale", "it", "Rendilo più formale"),
    ("um make this more formal", None, "Make this more formal"),
    ("eh hazlo más formal nueva línea por favor", "es", "Hazlo más formal nueva línea por favor"),
    ("ehm", "es", "ehm"),                   # nothing left: the spoken words are kept rather than an empty order
    ("", "es", ""), (None, "es", ""),
])
def test_command_instruction_is_cleaned_with_its_own_language(spoken, language, want):
    assert command.clean_instruction(spoken, language) == want


def test_command_instruction_language_is_detected_when_not_given():
    assert command.clean_instruction("eh pon esto más formal para el jefe de la empresa") == \
        "Pon esto más formal para el jefe de la empresa"


def test_fake_cleaner_runs_the_real_rules_for_every_language():
    fc = fakes.FakeCleaner(delay_s=0)
    fc.load()
    assert fc.clean("eh tengo veinticinco años", language="es", smart=True, tone="casual") == "Tengo 25 años"
    assert fc.clean("ähm das kostet zwanzig euro", language="de", smart=True) == "Das kostet 20 €."
    assert fc.clean("ehm myślę że powinniśmy jutro wyjechać", language="pl") == "Myślę że powinniśmy jutro wyjechać."
    assert fc.clean("um i think we should go", language="en") == "I think we should go."
    assert fc.clean("eh vamos a la playa mañana por la tarde con mis amigos", language="auto") == \
        "Vamos a la playa mañana por la tarde con mis amigos."
