from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # for consistent results


def is_english(text: str) -> bool:
    cleaned_text = text.replace("\n", "")
    try:
        return detect(cleaned_text) == "en"
    except:
        return False

def choose_real_text(text1: str, text2: str) -> str:
    """Choose the more likely real text between two options using language detection."""
    is_text1_english = is_english(text1)
    is_text2_english = is_english(text2)
    
    if is_text1_english and not is_text2_english:
        return text1
    return text2