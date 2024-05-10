from evaluate import load

def bert_score(predictions, references):

    bertscore = load("bertscore")
    bertscore_results = results = bertscore.compute(predictions=predictions, references=references, lang="ko")

    return bertscore_results.f1