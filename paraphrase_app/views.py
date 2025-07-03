from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .model_loader import tokenizer, model, label_encoder, MAX_SEQ_LENGTH
from django.conf import settings
import google.generativeai as genai
import numpy as np
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
def preprocess(sentence1, sentence2):
    tokens_1 = tokenizer(sentence1, truncation=True, padding='max_length', max_length=MAX_SEQ_LENGTH, return_tensors='tf')
    tokens_2 = tokenizer(sentence2, truncation=True, padding='max_length', max_length=MAX_SEQ_LENGTH, return_tensors='tf')

    return {
        "input_ids_1": tokens_1["input_ids"],
        "attention_mask_1": tokens_1["attention_mask"],
        "input_ids_2": tokens_2["input_ids"],
        "attention_mask_2": tokens_2["attention_mask"],
    }

@csrf_exempt
def api_classify(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        sentence1 = data.get('sentence1', '')
        sentence2 = data.get('sentence2', '')
        
        inputs = preprocess(sentence1, sentence2)
        probs = model.predict(inputs)
        pred_class = np.argmax(probs, axis=1)[0]
        result_type = label_encoder.inverse_transform([pred_class])[0]
        confidence = float(probs[0][pred_class])  

        
        return JsonResponse({
            'paraphrase_type': result_type,
            'confidence': confidence,
            'explanation': f'The sentences show {result_type.lower()} paraphrasing with {confidence*100:.1f}% confidence.'
        })
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)


# Configure Gemini API
genai.configure(api_key=settings.GEMINI_API_KEY)

@csrf_exempt
def generate_paraphrase_with_gemini(prompt_text):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"Only give Paraphrased version of the following sentence in Urdu:\n\n{prompt_text}"
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API error: {e}")
        return prompt_text + " (paraphrased)"
    

@csrf_exempt
def api_paraphrase(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        original_sentence = data.get('sentence', '')

        # Generate paraphrase using Gemini
        generated_paraphrase = generate_paraphrase_with_gemini(original_sentence)

        # Predict paraphrase type using your model
        inputs = preprocess(original_sentence, generated_paraphrase)
        probs = model.predict(inputs)
        pred_class = np.argmax(probs, axis=1)[0]
        confidence = float(probs[0][pred_class])
        result_type = label_encoder.inverse_transform([pred_class])[0]

        return JsonResponse({
            'original': original_sentence,
            'paraphrased': generated_paraphrase,
            'paraphrase_type': result_type,
            'confidence': confidence,
            'explanation': f'The paraphrased sentence shows {result_type.lower()} paraphrasing with {confidence*100:.1f}% confidence.'
        })

    return JsonResponse({'error': 'Invalid request method'}, status=405)
"""
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
"""