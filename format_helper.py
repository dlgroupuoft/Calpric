########################################### preprocessing helper functions ###########################################
import re
from langdetect import detect


def check_printable(text_list):
    new_texts = []
    for text_str in text_list:
        filtered_text = ''.join(char for char in text_str if not repr(char).startswith("'\\x"))
        new_texts.append(filtered_text)
    return new_texts


def contain_three_or_more_http(sent):
    sent = str(sent)
    keyword = 'http : / /'
    counts = sent.count(keyword)
    if counts >= 3:
        print("multiple websites found in the sent: {} times!".format(counts))
        print(sent)
        return True
    else:
        return False


def check_english(sent):
    if contain_three_or_more_http(sent):
        return False
    try:
        lang = detect(sent)
    except Exception:
        print("error in check_english(): detection failure!")
        print(sent)
        print("error in check_english(): detection failure!")
        print(sent)
        return False
    if lang == 'en':
        return True
    else:
        print("lang: " + str(lang))
        print("in check_english(): detected non-english")
        print(sent)
        print("in check_english(): detected non-english")
        print(sent)
        return False


def convert_abbrs(sent):
    sent = re.sub(r"ain(\'|\s){0,}t ", " is not ", sent)
    sent = re.sub(r"aren(\'|\s){0,}t ", " are not ", sent)
    sent = re.sub(r"couldn(\'|\s){0,}t ", " could not ", sent)
    sent = re.sub(r"didn(\'|\s){0,}t ", " did not ", sent)
    sent = re.sub(r"doesn(\'|\s){0,}t ", " does not ", sent)
    sent = re.sub(r"hadn(\'|\s){0,}t ", " had not ", sent)
    sent = re.sub(r"hasn(\'|\s){0,}t ", " has not ", sent)
    sent = re.sub(r"haven(\'|\s){0,}t ", " have not ", sent)
    sent = re.sub(r"isn(\'|\s){0,}t ", " is not ", sent)
    sent = re.sub(r"mightn(\'|\s){0,}t ", " might not ", sent)
    sent = re.sub(r"mustn(\'|\s){0,}t ", " must not ", sent)
    sent = re.sub(r"needn(\'|\s){0,}t ", " need not ", sent)
    sent = re.sub(r"shan(\'|\s){0,}t ", " shall not ", sent)
    sent = re.sub(r"shouldn(\'|\s){0,}t ", " should not ", sent)
    sent = re.sub(r"wasn(\'|\s){0,}t ", " was not ", sent)
    sent = re.sub(r"weren(\'|\s){0,}t ", " were not ", sent)
    sent = re.sub(r"won(\'|\s){0,}t ", " will not ", sent)
    sent = re.sub(r"wouldn(\'|\s){0,}t ", " would not ", sent)
    sent = re.sub(r"don(\'|\s){0,}t ", " do not ", sent)
    sent = re.sub(r"can(\'|\s){0,}t ", " cannot ", sent)
    sent = re.sub(r"mayn(\'|\s){0,}t ", " may not ", sent)

    sent = re.sub(r"\'m", " am ", sent)
    sent = re.sub(r"\'re", " are ", sent)
    sent = re.sub(r"\'ve", " have ", sent)
    sent = re.sub(r"\'ll", " will ", sent)
    sent = re.sub(r" ll ", " will ", sent)
    sent = re.sub(r"\'d", " would ", sent)
    sent = re.sub(r"\'s", " 's ", sent)

    sent = re.sub(r"Mr\.", "Mr ", sent)
    sent = re.sub(r"Ms\.", "Ms ", sent)
    sent = re.sub(r"Mrs\.", "Mrs ", sent)
    sent = re.sub(r"Dr\.", "Doctor ", sent)
    sent = re.sub(r"Prof\.", "Professor ", sent)
    sent = re.sub(r"Sr\.", "Senior ", sent)

    sent = re.sub(r"e\.g\.", "examples: ", sent)
    sent = re.sub(r"e\.g", "examples: ", sent)
    sent = re.sub(r"i\.e", "examples: ", sent)
    sent = re.sub(r"i\.e\.", "examples: ", sent)
    return sent


def convert_ip_addr(sent):
    sent = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ip address ", sent)
    return sent


def convert_website(sent):
    sent = re.sub \
            (
            r"(([\da-zA-Z])([_\w-]{,62})\.){,127}(([\da-zA-Z])[_\w-]{,61})?([\da-zA-Z]\.((xn\-\-[a-zA-Z\d]+)|([a-zA-Z]{2,})))",
            " website ", sent)
    return sent


def convert_phone_number(sent):
    sent = re.sub(r"(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})",
                  " phone number ", sent)
    return sent


def convert_email_addresses(sent):
    sent = re.sub(r"[^@\s]+@[^@\s]+\.[^@\s]+", " email address ", sent)
    return sent


def convert_unicode(sent):
    sent = re.sub(u"\u00a9", " ", sent)  # ©
    sent = re.sub(u"\u24ea", " , ", sent)  # ⓪
    sent = re.sub(u"\u2460", " , ", sent)  # ①
    sent = re.sub(u"\u2461", " , ", sent)  # ②
    sent = re.sub(u"\u2462", " , ", sent)  # ③
    sent = re.sub(u"\u2463", " , ", sent)  # ④
    sent = re.sub(u"\u0026", " and ", sent)  # &
    sent = re.sub(u"\u200b", "", sent)  # zero width space
    sent = re.sub(u"\\u2019s", " 's ", sent)  # ’s

    return sent


def convert_punct(sent):
    sent = re.sub(r"((^|\s|\()[A-Za-z0-9]\))", " , ", sent)
    sent = re.sub(r"^\d[\.\d]{0,}\.", " ", sent)
    sent = re.sub(r"\s\d[\.\d]{0,}\)", " , ", sent)
    sent = re.sub(r"\s\d{1,}\.", " ", sent)
    sent = re.sub(r"\s\(\d{1,}\)\s", " ", sent)
    sent = re.sub(r"\,", " , ", sent)
    sent = re.sub(r"\/", " / ", sent)
    sent = re.sub(r"\"", " \" ", sent)
    sent = re.sub(r"\[", " [ ", sent)
    sent = re.sub(r"\]", " ] ", sent)
    sent = re.sub(r"\(", " ( ", sent)
    sent = re.sub(r"\)", " ) ", sent)
    sent = re.sub(r"\?", " ? ", sent)
    sent = re.sub(r"\.", " . ", sent)
    sent = re.sub(r"\:", " : ", sent)
    sent = re.sub(r"\!", " . ", sent)
    return sent


def convert_sent(sent):
    sent = convert_abbrs(sent)
    sent = convert_ip_addr(sent)
    sent = convert_email_addresses(sent)
    sent = convert_website(sent)
    sent = convert_phone_number(sent)
    sent = convert_unicode(sent)
    sent = convert_punct(sent)

    sent = re.sub('(\n|\t|\s)+', ' ', sent.strip())
    return sent
