import evaluate

def rouge_score(predictions, references):
    rouge = evaluate.load('rouge')
    
    #predictions = ["hello there", "general kenobi"]
    #references = ["hello there", "general kenobi"]
    
    results = rouge.compute(predictions=predictions, references=references)
    return results