import evaluate

def rouge_score(predictions, references):
    rouge = evaluate.load('rouge')
    
    results = rouge.compute(predictions=predictions, references=references)

    '''
    # added
    precision = results['rouge1'].precision
    recall = results['rouge1'].recall
    f1_score = results['rouge1'].fmeasure
    
    return precision, recall, results, f1_score
    '''
    return results
