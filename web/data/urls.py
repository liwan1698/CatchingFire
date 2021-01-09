from django.urls import re_path, include, path
from data.views import Classify, ClassifyTags, classify_stats


urlpatterns = [
    re_path('^classify/$', Classify.as_view(), name='classify'),
    re_path('^tags/$', ClassifyTags.as_view(), name='tags'),
    re_path('^status/$', classify_stats, name='status'),
]
