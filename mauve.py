from evaluate import load

def mauve_score(predictions, references):

    mauve = load('mauve')
    mauve_results = mauve.compute(predictions=predictions, references=references)

    return mauve_results.mauve 