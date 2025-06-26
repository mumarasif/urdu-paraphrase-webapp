from django.urls import path
from . import views

app_name = 'paraphrase_app'

urlpatterns = [
    path('', views.home, name='home'),
    path('classify/', views.classify, name='classify'),
    path('paraphrase/', views.paraphrase, name='paraphrase'),
    path('eda/', views.eda, name='eda'),
    path('api/classify/', views.api_classify, name='api_classify'),
    path('api/paraphrase/', views.api_paraphrase, name='api_paraphrase'),
]
