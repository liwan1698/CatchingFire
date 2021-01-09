from django.urls import re_path, include, path
from model import views


urlpatterns = [
    re_path('^classify_train_all/$', views.classify_train_all, name='classify_train_all'),
]
