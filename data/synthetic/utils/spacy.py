import spacy


SPATIAL_TOKENS = {
    'left', 'right', 'top', 'bottom', 'upper', 'lower', 'middle', 'center', 'central',
    'corner', 'edge', 'border',
    'front', 'back', 'side', 'middle', 'mid',
    'first', 'second', 'third', 'fourth', 'fifth',
    'last', 'next'
}

nlp = spacy.load("en_core_web_sm") 

def has_spatial_expression(noun: str) -> bool:
    """
    判断 noun 是否包含空间/方位描述词。
    方法：
    1. 检查是否有词在 spatial_tokens 中
    2. 检查是否有形容词修饰名词且语义为空间
    """
    doc = nlp(noun.lower())
    words = [token.text for token in doc]
    if set(words) & SPATIAL_TOKENS:
        return True

    for token in doc:
        if token.dep_ == "amod" and token.text in SPATIAL_TOKENS:
            return True

    for token in doc:
        if token.dep_ == "compound" and any(t in SPATIAL_TOKENS for t in token.text.split('-')):
            return True

    for token in doc:
        if token.dep_ == "prep" and token.text in ['in', 'on', 'at']:
            for child in token.children:
                if child.text in SPATIAL_TOKENS:
                    return True

    return False

if __name__ == "__main__":
    test_nouns = [
        "The man on left in black shirt"
    ]
    for noun in test_nouns:
        print(f"'{noun}' has spatial expression: {has_spatial_expression(noun)}")