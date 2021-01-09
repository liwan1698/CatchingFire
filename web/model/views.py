import logging
from rest_framework.decorators import api_view
from django.http import JsonResponse


@api_view(['GET'])
def classify_train_all(request):
    """
    全量训练文本分类模型
    :param request:
    :return:
    """
    dataset = request.GET.get("dataset")
    algorithm = request.GET.get("algorithm")


    return JsonResponse({"code": 200}, json_dumps_params={'ensure_ascii': False})

