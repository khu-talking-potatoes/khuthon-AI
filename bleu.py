import evaluate

def bleu_score(predictions, references):

    bleu = evaluate.load("bleu")

    results = bleu.compute(predictions=predictions, references=references)
    
    bleu = results['bleu']
    
    return bleu 