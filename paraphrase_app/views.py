from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .model_loader import tokenizer, model, label_encoder, MAX_SEQ_LENGTH
from django.conf import settings
import google.generativeai as genai
import numpy as np
import json
import random
import pandas as pd
import os
import tempfile
from datetime import datetime

def home(request):
    return render(request, 'paraphrase_app/home.html')

def classify(request):
    return render(request, 'paraphrase_app/classify.html')

def paraphrase(request):
    return render(request, 'paraphrase_app/paraphrase.html')

def document_prediction(request):
    return render(request, 'paraphrase_app/document_prediction.html')

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

# Global variable to store the latest processed results for download
latest_results_df = None

@csrf_exempt
def api_document_prediction(request):
    global latest_results_df
    
    if request.method == 'POST':
        if 'document' not in request.FILES:
            return JsonResponse({'error': 'No document file provided'}, status=400)
        
        document_file = request.FILES['document']
        file_extension = os.path.splitext(document_file.name)[1].lower()
        
        try:
            # Read the file based on its extension
            if file_extension == '.csv':
                df = pd.read_csv(document_file)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(document_file)
            else:
                return JsonResponse({'error': 'Unsupported file format. Please upload CSV or Excel file.'}, status=400)
            
            # Check if required columns exist
            if 'Sentence1' not in df.columns or 'Sentence2' not in df.columns:
                return JsonResponse({'error': 'File must contain columns named "Sentence1" and "Sentence2"'}, status=400)
            
            # Process each row for prediction
            results = []
            for _, row in df.iterrows():
                sentence1 = row['Sentence1']
                sentence2 = row['Sentence2']
                
                # Skip empty rows
                if pd.isna(sentence1) or pd.isna(sentence2) or sentence1.strip() == '' or sentence2.strip() == '':
                    continue
                
                # Predict paraphrase type
                inputs = preprocess(sentence1, sentence2)
                probs = model.predict(inputs)
                pred_class = np.argmax(probs, axis=1)[0]
                confidence = float(probs[0][pred_class])
                result_type = label_encoder.inverse_transform([pred_class])[0]
                
                results.append({
                    'sentence1': sentence1,
                    'sentence2': sentence2,
                    'paraphrase_type': result_type,
                    'confidence': confidence
                })
            
            # Create a DataFrame with results
            results_df = pd.DataFrame(results)
            
            # Store the results for download
            latest_results_df = pd.DataFrame({
                'Sentence1': results_df['sentence1'],
                'Sentence2': results_df['sentence2'],
                'Paraphrase_Type': results_df['paraphrase_type'],
                'Confidence': results_df['confidence']
            })
            
            # Return preview of results (first 10 rows)
            preview = results[:10] if len(results) > 10 else results
            
            return JsonResponse({
                'success': True,
                'total_rows': len(results),
                'preview': preview
            })
            
        except Exception as e:
            return JsonResponse({'error': f'Error processing file: {str(e)}'}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def download_results(request):
    global latest_results_df
    
    if latest_results_df is None:
        return HttpResponse('No results available for download', status=404)
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
        # Write the DataFrame to Excel
        latest_results_df.to_excel(tmp.name, index=False, engine='openpyxl')
        tmp_path = tmp.name
    
    # Read the file and create the response
    with open(tmp_path, 'rb') as f:
        file_data = f.read()
    
    # Delete the temporary file
    os.unlink(tmp_path)
    
    # Create the response with appropriate headers
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    response = HttpResponse(
        file_data,
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = f'attachment; filename=paraphrase_results_{timestamp}.xlsx'
    
    return response

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