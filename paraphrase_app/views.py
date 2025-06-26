from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import random

def home(request):
    return render(request, 'paraphrase_app/home.html')

def classify(request):
    return render(request, 'paraphrase_app/classify.html')

def paraphrase(request):
    return render(request, 'paraphrase_app/paraphrase.html')

def eda(request):
    # Sample data for EDA visualizations
    context = {
        'class_distribution': {
            'labels': ['Lexical', 'Syntactic', 'Semantic', 'Morphological', 'Compound', 'Phrasal', 'Structural', 'Contextual', 'Stylistic', 'Discourse', 'Pragmatic', 'Temporal', 'Modal', 'Negation'],
            'data': [145, 132, 128, 98, 87, 76, 65, 58, 52, 48, 43, 38, 35, 28]
        },
        'length_stats': {
            'labels': ['1-10', '11-20', '21-30', '31-40', '41-50', '50+'],
            'data': [45, 128, 234, 189, 87, 32]
        }
    }
    return render(request, 'paraphrase_app/eda.html', context)

@csrf_exempt
def api_classify(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        sentence1 = data.get('sentence1', '')
        sentence2 = data.get('sentence2', '')
        
        # Mock classification result
        paraphrase_types = ['Lexical', 'Syntactic', 'Semantic', 'Morphological', 'Compound', 'Phrasal']
        result_type = random.choice(paraphrase_types)
        confidence = round(random.uniform(0.7, 0.95), 3)
        
        return JsonResponse({
            'paraphrase_type': result_type,
            'confidence': confidence,
            'explanation': f'The sentences show {result_type.lower()} paraphrasing with {confidence*100:.1f}% confidence.'
        })
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def api_paraphrase(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        original_sentence = data.get('sentence', '')
        
        # Mock paraphrase generation
        paraphrases = [
            original_sentence.replace('ہے', 'موجود ہے'),
            original_sentence + ' کا مطلب یہ ہے',
            original_sentence.replace('اور', 'تھا اور'),
        ]
        
        generated_paraphrase = random.choice(paraphrases) if paraphrases else original_sentence + ' (paraphrased)'
        
        # Mock classification of the pair
        paraphrase_types = ['Lexical', 'Syntactic', 'Semantic', 'Morphological']
        result_type = random.choice(paraphrase_types)
        confidence = round(random.uniform(0.7, 0.95), 3)
        
        return JsonResponse({
            'original': original_sentence,
            'paraphrased': generated_paraphrase,
            'paraphrase_type': result_type,
            'confidence': confidence
        })
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)
