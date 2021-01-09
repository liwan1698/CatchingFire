import logging
from rest_framework.decorators import api_view
from django.http import JsonResponse

from data.models import ClassifyData
from model.classify.fasttext import FastText


@api_view(['GET'])
def classify_train_all(request):
    """
    全量训练文本分类模型
    :param request:
    :return:
    """
    dataset = request.GET.get("dataset")
    algorithm = request.GET.get("algorithm")
    fasttext = FastText()
    fasttext.build()
    fasttext.train()
    result = fasttext.predict("马晓旭意外受伤让国奥警惕 无奈大雨格外青睐殷家军记者傅亚雨沈阳报道")

    return JsonResponse({"code": 200}, json_dumps_params={'ensure_ascii': False})

