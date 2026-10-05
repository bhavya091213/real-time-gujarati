import pytest

from gujusub.fillers import clean_event, clean_text


@pytest.mark.parametrize(
    "text, expected",
    [
        ("um hello there", "hello there"),
        ("hello um there", "hello there"),
        ("hello there um", "hello there"),
        ("Um, hello", "hello"),
        ("Uh... hello", "hello"),
        ("અં મારું નામ ભવ્ય છે", "મારું નામ ભવ્ય છે"),
        ("મારું હં નામ", "મારું નામ"),
        ("", ""),
        ("um", ""),
        ("um uh hmm", ""),
    ],
)
def test_hesitations_always_dropped(text, expected):
    assert clean_text(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I like this song", "I like this song"),
        ("so that is why", "that is why"),
        ("this is so good", "this is so good"),
        ("like I was saying", "I was saying"),
        ("um like I was saying", "I was saying"),
        ("it was like like huge", "it was like huge"),
        ("you know it was good", "it was good"),
        ("I mean it was good", "it was good"),
        ("do you know him", "do you know him"),
        ("એટલે મારું નામ", "મારું નામ"),
        ("એટલે કે મારું નામ", "મારું નામ"),
    ],
)
def test_contextual_fillers_position_dependent(text, expected):
    assert clean_text(text) == expected


def test_trailing_contextual_filler_only_dropped_on_final():
    assert clean_text("that is why so") == "that is why so"
    assert clean_text("that is why so", final=True) == "that is why"
    assert clean_text("that is why so um", final=True) == "that is why"


def test_clean_event_keeps_committed_tail_split():
    assert clean_event("um hello", "there um") == ("hello", "there")


def test_clean_event_phrase_across_boundary():
    assert clean_event("well you", "know it was") == ("", "it was")


def test_clean_event_empty():
    assert clean_event("", "") == ("", "")
    assert clean_event("um", "") == ("", "")


# --- English captions (unit 3.2) -------------------------------------------

@pytest.mark.parametrize("opener", ["So,", "so", "Okay,", "OK", "Right,"])
def test_english_sentence_initial_marker_kept_when_followed_by_words(opener):
    text = f"{opener} today we talk about kindness"
    assert clean_event(text, "", final=True, lang="en") == (text, "")


def test_english_initial_marker_kept_after_dropped_hesitation():
    assert clean_event("um so we begin", "", lang="en") == ("so we begin", "")


def test_english_standalone_marker_still_dropped():
    assert clean_event("So,", "", final=True, lang="en") == ("", "")
    assert clean_event("so um", "", lang="en") == ("", "")


def test_english_hesitations_dropped():
    assert clean_event("we um begin", "uh now", lang="en") == ("we begin", "now")


def test_english_other_initial_fillers_unchanged():
    assert clean_event("like I said", "", lang="en") == ("I said", "")


def test_gujarati_default_still_drops_initial_so():
    assert clean_event("so today we talk", "") == ("today we talk", "")
    assert clean_event("so today we talk", "", lang="gu") == ("today we talk", "")
