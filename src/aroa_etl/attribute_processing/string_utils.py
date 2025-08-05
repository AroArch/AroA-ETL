import pandas as pd
import re

# Preprocessing of data based on a skript from Uwe Ossenberg
ascii_replacements = {
        'á': 'a',        'ï': 'i',        'ş': 's',        'ó': 'o',
        'ł': 'l',        'ñ': 'n',        'è': 'e',        'ç': 'c',
        'ß': 'ss',        'ô': 'o',        'ü': 'u',        'æ': 'ae',
        'ø': 'o',        'û': 'u',        'ã': 'a',        'ê': 'e',
        'ë': 'e',        'ù': 'u',        'ï': 'i',        'î': 'i',
        'é': 'e',        'í': 'i',        'ú': 'u',        'ý': 'y',
        'à': 'a',        'ì': 'i',        'ò': 'o',        'ã': 'a',
        'ñ': 'n',        'õ': 'o',        'ç': 'c',        'ă': 'a',
        'ā': 'a',        'ē': 'e',        'ī': 'i',        'ō': 'o',
        'ū': 'u',        'ȳ': 'y',        'ǎ': 'a',        'ě': 'e',
        'ǐ': 'i',        'ǒ': 'o',        'ǔ': 'u',        'ǜ': 'u',
        'ǽ': 'ae',       'ð': 'd',        'œ': 'oe',       'ẽ': 'e',
        'ỹ': 'y',        'ũ': 'u',        'ȩ': 'e',        'ȯ': 'o',
        'ḧ': 'h',        'ẅ': 'w',        'ẗ': 't',        'ḋ': 'd',
        'ẍ': 'x',        'ẁ': 'w',        'ẃ': 'w',        'ỳ': 'y',
        'ÿ': 'y',        'ỹ': 'y',        'ŷ': 'y',        'ą': 'a',
        'į': 'i',        'ś': 's',        'ź': 'z',        'ć': 'c',
        'ń': 'n',        'ę': 'e',        'ţ': 't',        'ģ': 'g',
        'ķ': 'k',        'ņ': 'n',        'ļ': 'l',        'ż': 'z',
        'ċ': 'c',        'š': 's',        'ž': 'z',        'ď': 'd',
        'ľ': 'l',        'ř': 'r',        'ǧ': 'g',        'ǳ': 'dz',
        'ǆ': 'dz',     'ǉ': 'lj',       'ǌ': 'nj',       'ǚ': 'u',
        'ǘ': 'u',        'ǜ': 'u',        'ǟ': 'a',        'ǡ': 'a',
        'ǣ': 'ae',       'ǥ': 'g',        'ǭ': 'o',        'ǯ': 'z',
        'ȟ': 'h',        'ȱ': 'o',        'ȹ': 'y',        'ḭ': 'i',
        'ḯ': 'i',        'ḱ': 'k'
    }

umlaut_replacements = {
    'ä': 'a',    'ae': 'a',
    'ö': 'o',    'oe': 'a',
    'ü': 'u',    '(?<!a)ue': 'a',
}

phonetic_bigram_replacements = {
    'th': 't',
    'ck': 'k',
    'ph': 'f',
    'w': 'v',
    'y': 'i',
    'j': 'i',
    'tz': 'z',
}

phonetic_replacements = {
    **phonetic_bigram_replacements, 
    **umlaut_replacements,
    **ascii_replacements
}

visual_num_to_char = {
     # numbers
    '0' : 'O',      '1' : 'l',      '2': '',        '3' : 'B',      '4' : 'A',     '5' : '',
    '6': '',        '7' : 'T',      '8': 'B',       '9' : '',
}

visual_num_to_num = {
     # numbers
    '0' : '0',      '1' : '1',      '2': '2',        '3' : '3',      '4' : '4',     '5' : '5',
    '6': '6',        '7' : '7',      '8': '8',       '9' : '9',
}

visual_german_replacements = {
     # Letters
    'A' : 'A',      'B' : 'B',      'C' : 'C',      'D' : 'D',      'E' : 'E',     'F' : 'F',
    'G' : 'G',      'H' : 'H',      'I' : 'I',      'J' : 'J',      'K' : 'K',     'L' : 'L',
    'M' : 'M',      'N' : 'N',      'O' : 'O',      'P' : 'P',      'Q' : 'Q',     'R' : 'R',
    'S' : 'S',      'T' : 'T',      'U' : 'U',      'V': 'V',       'W' : 'W',     'X' : 'X',
    'Y' : 'Y',      'Z' : 'Z',
    # Lowercase letters
    'a' : 'a',      'b' : 'b',      'c' : 'c',      'd': 'd',       'e' : 'e',     'f' : 'f',
    'g' : 'g',      'h' : 'h',      'i' : 'i',      'j': 'j',       'k' : 'k',     'l' : 'l',
    'm' : 'm',      'n' : 'n',      'o' : 'o',      'p' : 'p',      'q' : 'q',     'r' : 'r',
    's' : 's',      't' : 't',      'u' : 'u',      'v': 'v',       'w' : 'w',     'x' : 'x',
    'y' : 'y',      'z' : 'z',
    # other
    'ß' : 'ß',      'ä' : 'ä',      'Ä' : "Ä",      'ö' : 'ö',      'Ö' : 'Ö',      'ü' : 'ü', 
    'Ü' : 'Ü',
}
visual_symbol_replacements = {
    # Specials
    '-' : '-',      ' ' : ' ',      '|' : '',      '!' : '',       '#' : '',      '$' : '',
    '%' : '',       '&' : '',       '+' : '',       '*' : '',       '@' : '',      '?' : '',
    ';' : '',       ',' : '',       '°' : '',       '~' : '',       '*' : '',      
    '@' : '', 
}
visual_non_ascii_replacements = {
    # override unidecode
    'р' : 'p',      'п' : 'n',      'Η' : 'H',      'ή' : 'n',      'Γ' : 'T',      'ѕ' : 's',
    'м' : 'M',      'д' : 'A',      'σ' : 'o',      'Д' : 'A',      'ш' : 'w',      'с' : 'c',
    'ω' : 'w',      'Θ' : 'O',      'ъ' : 'b',      'Π' : 'N',      'ن' : 'u',      'β' : 'ß',
    'х' : 'x',      'ง' : 'i',       'μ' : 'u',     'ь' : 'b',      'ц' : 'u',      'в' : 'B',
    'И' : 'N',      'С' : 'C',      'ρ' : 'p',      'л' : 'n',      'ы' : 'bl',     'З' : 'B',
    'г' : 'r',      'В' : 'B',      'ə' : '',       'Л' : 'N',      'Ρ' : 'P',      'я' : '',
    '○' : 'o',      'א' : 'x',      'у' : 'y',      'Ь' : 'b',      'い' : 'w',     'π' : 'n',
    'й' : 'N',      'ه' : 'o',      'Ы' : 'bl',     'Х' : 'X',      '1' : 'l',      'н' : 'H',
    'У' : 'y',      'ж' : 'x',      'מ' : 'n',      'Г' : 'T',      'م' : 'p',      'γ' : 'y',
    'Λ' : 'A',      '。' : '.',      'ν' : 'v',     '•' : '',       'โ' : 'l',       'η' : 'n',
    'ת' : 'n',      'Š': 'S',       'm': 'm',       'Н': 'H',       'ด': 'd',       'δ': 'd',
    'Ψ': 'Ps',      'キ': 'ki',     'č': 'c',       'ı': 'i',       'ς': 's',       'ň': 'n',
    'ê': 'e',       'わ': 'wa',     'ę': 'e',       'ř': 'r',       'Б': 'B',
    'よ': 'yo',     'ة': 'ö',       'ó': 'o',       'φ': 'p',       'χ': 'x',     'í': 'i',
    'І': 'I',       'Σ': 'S',      'ち': 't',       'Č': 'C',       'ė': 'e',      'ń': 'n',
    'М': 'M',       'ψ': 'y',       'ί': 'i',       'ق': 'q',       '工': 'I',
    'υ': 'u',       'ả': 'a',      'ź': 'z',        'す': 'T',       'λ': 'l',
    'е': 'e',       'Ά': 'A',       'Ñ': 'N',       'É': 'E',
    'θ': 'O',      'ť': 't',        'Ø': 'O',       'Ј': 'J',
    'а': 'a',       'ë': 'e',       'り': 'n',      'κ': 'k',       'ε': 'e',
    'Ú': 'U',       'ě': 'e',       'د': 'i',       'ằ': 'a',       'Ζ': 'Z',
    'Ν': 'N',      'ひ': 'U',       'П': 'N',        'ć': 'c',      'ũ': 'u',
    'Т': 'T',       'ス': 'J',      'Э': 'E',        'ј': 'j',      'ů': 'u',
    'о': 'o',       'О': 'O',       'ą': 'a',       'Û': 'U',       'Á': 'A',      'ξ': 'E',
    'Ό': 'O',       'æ': 'ae',      'и': 'n',       'ч': 'y',
    'ă': 'a',       'さ': 't',       'お': 'F',     
    'ż': 'z',       'Έ': 'E',       'ł': 'l',       'Ο': 'O',        'ン': 'y',      '年': 'T',
    'ό': 'o',       'à': 'a',       'Χ': 'X',        
    '்': '',        'Ż': 'Z',       'Ş': 'S',       'ص': 'u',
    'ה': 'h',       'Р': 'P',       'ム': 'A',       'ت': 'ü',       'ά': 'a',
    'á': 'a',       'ま': 'L',      'ô': 'o',        'è': 'e',       
    'к': 'k',       'で': 'T',      'é': 'e',        'έ': 'e',       'Ι': 'I',
    'ο': 'o',       'і': 'i',       'Ł': 'L',       'ú': 'u',      
    'α': 'a',       'Δ': 'D',       'Ε': 'E',
    'っ': 's',      'ア': 'y',       'т': 't',       'ã': 'a',
     'ő': 'ö',      'ι': 'i',       'Е': 'E',       'Ω': 'O',        'Ś': 'S',       
     'ñ': 'n',      'Τ': 'T',       'К': 'K',       'ョ': 'z',        'š': 's',       'ý': 'y',
     'Β': 'B',      '日': 'B ',      'ş': 's',      
     'ا': '',       'ž': 'z',       'з': 'z',       'ç': 'c',       'Κ': 'K',
     'خ': 'i',     '下': 'T',       'Ž': 'Z',       'Μ': 'M',
     'å': 'a',      'ζ': 'z',       'А': 'A',       'â': 'a',       'б': 'b',       '̇': '',
     'れ': 'h',     'じ': 'i',      'の': 'D',       '.': '.',     
     'τ': 't',      "'": "",      
     'Α': 'A',       'ś': 's',      'ら': 'b',     
}

visual_replacements = {**visual_num_to_char, **visual_german_replacements, **visual_symbol_replacements, **visual_non_ascii_replacements }

def fix_visual_character_decoding(string):
    string = str(string)
    if len(string) == 0 or len([c for c in string if c in visual_non_ascii_replacements])/len(string)>0.3:
        return ""
    new_string = [
        visual_replacements[char] 
        for char in string
        if char in visual_replacements
    ]
    return "".join(new_string)

def fix_name_uppercasing(string):
    for word_occurrence in re.finditer("[a-zA-Zäüöß]+",string):
        start, end = word_occurrence.span()
        word = word_occurrence.group()
        fixed_word = f"{word[0].upper()}{word[1:].lower()}"
        string = f"{string[:start]}{fixed_word}{string[end:]}"
    return string

def replace_special_character(name: str):
    for pattern,replace in ascii_replacements.items():
        name = re.sub(pattern,replace,name)
    return name

def replace_umlaut_character(name: str):
    for pattern,replace in umlaut_replacements.items():
        name = re.sub(pattern,replace,name)
    return name

def replace_phonetic_character(name: str):
    for pattern,replace in phonetic_bigram_replacements.items():
        name = re.sub(pattern,replace,name)
    return name

def remove_double_characters(name:str):
    # Use regex to replace consecutive double characters with single characters
    return re.sub(r'([a-zA-Z])\1', r'\1', name)

def remove_lang_specific_last_name_endings(name):
    name = re.sub(r'owa$|ova$','',name)
    name = re.sub(r'sohns$','sons',name)
    name = re.sub(r'sohn$','son',name)
    name = re.sub(r'(?<=sk|ck)a$','i',name)
    return name

def remove_maiden_name(name):
    name = re.sub(r"(?P<maiden>\sgeb\.?\s.*)","",name)
    name = re.sub(r"(?P<maiden>\sgesch\.?\s.*)","",name)
    return name

def preprocess_name(name):
    name = name.lower()
    name = replace_special_character(name)
    name = replace_umlaut_character(name)
    name = replace_phonetic_character(name)
    name = remove_double_characters(name)
    return name

def preprocess_last_name(name):
    name = name.lower()
    name = remove_lang_specific_last_name_endings(name)
    name = remove_maiden_name(name)
    name = preprocess_name(name)
    return name
