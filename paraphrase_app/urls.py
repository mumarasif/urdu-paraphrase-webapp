from django.urls import path
from . import views

app_name = 'paraphrase_app'

urlpatterns = [
    path('', views.home, name='home'),
    path('classify/', views.classify, name='classify'),
    path('paraphrase/', views.paraphrase, name='paraphrase'),
    path('eda/', views.eda, name='eda'),
    path('document_prediction/', views.document_prediction, name='document_prediction'),
    path('api/classify/', views.api_classify, name='api_classify'),
    path('api/paraphrase/', views.api_paraphrase, name='api_paraphrase'),
    path('api/document_prediction/', views.api_document_prediction, name='api_document_prediction'),
    path('api/download_results/', views.download_results, name='download_results'),
]
