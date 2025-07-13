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
    # Comprehensive sample data for EDA visualizations
    total_tokens = 196254
    total_sentences = 10500
    unique_words = 12212
    urdu_words = ['کے', 'کی', 'میں', 'نے', 'کا', 'سے', 'کو', 'ہے', 'اور', 'پر']
    counts = [8243, 6262, 6129, 5239, 3519, 3099, 3057, 2990, 2713, 2412]
    
    context = {
        # Corpus Overview
        'corpus_stats': {
            'total_sentence_pairs': 5250,
            'total_sentences': total_sentences,
            'total_tokens': total_tokens,
            'unique_words': unique_words,
            'vocabulary_size': unique_words,
            'lexical_diversity': round(unique_words / total_tokens, 3),  # unique words / total tokens
            'word_density': round(total_tokens / total_sentences, 1)  # average tokens per sentence
        },
        
        # Paraphrase Type Distribution
        'class_distribution': {
            'labels': ['Inflectional changes', 'Derivational changes', 'Spelling and format changes', 'Same-polarity substitutions', 'Synthetic–analytic substitutions', 'Opposite-polarity substitutions', 'Diathesis alterations', 'Negation switching', 'Punctuation and format changes', 'Direct/Indirect style alterations', 'Semantic changes', 'Change of order', 'Addition/deletion of information', 'English to Urdu translation changes'],
            'data': [145, 118, 920, 633, 421, 200, 99, 81, 477, 248, 565, 253, 942, 148],
            'percentages': [ 4.1, 3.3,26.1, 18.0, 11.9, 5.7, 2.8, 2.3, 13.6, 7.0, 16.0, 7.2, 26.8, 4.2]
        },        

        # Sentence Length Analysis
        'length_stats': {
            'labels': json.dumps(['1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31+']),
            'data': json.dumps([300, 1383, 1153, 486, 389, 216, 1519]),
            'avg_length': 36,
            'min_length': 3,
            'max_length': 771
        },
        
        # Top Frequent Words
        'frequent_words_list': list(zip(urdu_words, counts)),  # for template loop
        'frequent_words': {
            'urdu_words': urdu_words,
            'counts': counts
        },
        
        # Stopword Statistics
        'stopword_stats': {
            'total_stopwords': 50616,
            'unique_stopwords': 187,
            'stopword_percentage': 25.8,
            'repeated_words': 7705
        },
        
        # Class Imbalance
        'class_imbalance': {
            'most_samples': {'type': 'Addition/deletion of information', 'count': 942},
            'least_samples': {'type': 'Negation switching', 'count': 81},
            'imbalance_ratio': 11.6  # most/least
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
def process_text_with_lughatnlp(text):
    try:
        from LughaatNLP import LughaatNLP
        urdu_text_processing = LughaatNLP()
        
        normalized_text = urdu_text_processing.normalize(text)
        filtered_text = urdu_text_processing.remove_stopwords(text)
        stemmed_sentence = urdu_text_processing.urdu_stemmer(text)
        tokens = urdu_text_processing.urdu_tokenize(text)
        
        return {
            'normalized': normalized_text,
            'filtered': filtered_text,
            'stemmed': stemmed_sentence,
            'tokens': tokens
        }
    except ImportError:
        # If LughaatNLP is not installed, return empty results
        return {
            'normalized': text,
            'filtered': text,
            'stemmed': text,
            'tokens': [text]
        }
    except Exception as e:
        # Handle any other exceptions
        print(f"Error processing text with LughaatNLP: {e}")
        return {
            'normalized': text,
            'filtered': text,
            'stemmed': text,
            'tokens': [text]
        }

@csrf_exempt
def api_classify(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        sentence1 = data.get('sentence1', '')
        sentence2 = data.get('sentence2', '')
        
        # Process text with LughaatNLP
        sentence1_processing = process_text_with_lughatnlp(sentence1)
        sentence2_processing = process_text_with_lughatnlp(sentence2)
        
        inputs = preprocess(sentence1, sentence2)
        probs = model.predict(inputs)
        pred_class = np.argmax(probs, axis=1)[0]
        result_type = label_encoder.inverse_transform([pred_class])[0]
        confidence = float(probs[0][pred_class])  
        
        return JsonResponse({
            'paraphrase_type': result_type,
            'confidence': confidence,
            'explanation': f'The sentences show {result_type.lower()} paraphrasing with {confidence*100:.1f}% confidence.',
            'sentence1_processing': sentence1_processing,
            'sentence2_processing': sentence2_processing
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