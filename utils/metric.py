import editdistance

answers = ['coca cola', 'coca cola company']
preds = ['the coca', 'cocacola', 'coca cola', 'cola', 'cat']

def ANLS(answers: list, predicted: str):
    max_s = 0
    for answer in answers:
        NL = editdistance.eval(answer, predicted) / max(len(answer), len(predicted))
        s = 1 - NL if NL < 0.5 else 0
        max_s = max(max_s, s)
    return max_s
