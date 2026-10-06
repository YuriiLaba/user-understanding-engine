import re


class LayoutDecoder:
    """Decodes text typed in wrong keyboard layout."""
    
    def __init__(self):
        self.ukr_mac_map = {
            'q': 'й', 'w': 'ц', 'e': 'у', 'r': 'к', 't': 'е', 'y': 'н', 'u': 'г',
            'i': 'ш', 'o': 'щ', 'p': 'з', '[': 'х', ']': 'ї',
            'a': 'ф', 's': 'і', 'd': 'в', 'f': 'а', 'g': 'п', 'h': 'р', 'j': 'о',
            'k': 'л', 'l': 'д', ';': 'ж', '\'': 'є',
            'z': 'я', 'x': 'ч', 'c': 'с', 'v': 'м', 'b': 'и', 'n': 'т', 'm': 'ь',
            ',': 'б', '.': 'ю', '/': '.', '`': '₴'
        }
    
    def decode(self, text: str, layout: str = 'ukr_mac') -> str:
        """Decode text, preserving tokens like <cmd+c>."""
        if layout != 'ukr_mac':
            return text
        
        # Split into tokens and text
        parts = re.split(r'(<[^>]+>)', text)
        decoded_parts = []
        
        for part in parts:
            if part.startswith('<') and part.endswith('>'):
                decoded_parts.append(part)
                continue
            
            translated = []
            for char in part:
                lower = char.lower()
                if lower in self.ukr_mac_map:
                    mapped = self.ukr_mac_map[lower]
                    translated.append(mapped.upper() if char.isupper() else mapped)
                else:
                    translated.append(char)
            
            decoded_parts.append("".join(translated))
        
        return "".join(decoded_parts)