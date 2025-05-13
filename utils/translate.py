from googletrans import Translator

translator = Translator()

def translate_answer(text, lang_choice):
    lang_map = {"Yoruba": "yo", "Hausa": "ha", "Igbo": "ig"}
    if lang_choice == "English":
        return text
    translated = translator.translate(text, dest=lang_map[lang_choice])
    return translated.text
