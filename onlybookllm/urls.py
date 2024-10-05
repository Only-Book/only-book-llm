from django.urls import path
from .views import *

urlpatterns = [
    # path("", main, name="main"),
    path('get_books/', get_books, name='get_books'),
    path('chatbot/', chatbot_response, name='chatbot_response'),
]