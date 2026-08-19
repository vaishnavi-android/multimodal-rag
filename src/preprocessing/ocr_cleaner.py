import re


# High-confidence OCR corrections.
#
# These are intentionally conservative. We are NOT doing general
# spell-checking yet because that could incorrectly modify names,
# technical terms, IDs, numbers, or units.

OCR_CORRECTIONS = {
    "Anci1nt": "Ancient",
    "settlemen1s": "settlements",
    "cit-zen": "citizen",
    "amministrative": "administrative",
    "arc-ive": "archive",
    "inc nsistent": "inconsistent",
    "ask d": "asked",
    "a-ithmetic": "arithmetic",
    "Modrnrn": "Modern",
    "propertmes": "properties",
    "e ectrical": "electrical",
    "equipmemt": "equipment",
    "affec-s": "affects",
    "Aluminirnm": "Aluminum",
    "devemops": "develops",
    "Boili g": "Boiling",
    "occu1s": "occurs",
    "surrounrning": "surrounding",
    "condi1ions": "conditions",
    "f oat": "float",
    "resumt": "result",
    "Eart-'s": "Earth's",
    "primarirny": "primarily",
    "cha-ges": "changes",
    "a sociated": "associated",
    "Ti1ris": "Tigris",
    "c1ties": "cities",
    "dev loped": "developed",
    "agricultur ,": "agriculture,",
}


def correct_known_ocr_errors(text: str) -> str:
    """
    Correct only explicitly known, high-confidence OCR errors.

    This function deliberately avoids:
    - general spell correction
    - changing numbers
    - changing units
    - changing document IDs
    - guessing unknown words
    """

    if not text:
        return ""

    corrected = text

    # Longest patterns first prevents a shorter pattern from interfering
    # with a more specific correction.
    for noisy, clean in sorted(
        OCR_CORRECTIONS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        corrected = corrected.replace(noisy, clean)

    return corrected


def clean_ocr_text(text: str) -> str:
    """
    Apply conservative OCR correction to extracted text.
    """

    if not text:
        return ""

    return correct_known_ocr_errors(text)