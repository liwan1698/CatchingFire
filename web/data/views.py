from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from django.views.generic.base import View
from django.http import JsonResponse
from data.models import ClassifyData, ClassifyTag
import logging
import json


class Classify(View):
    def get(self, request):
        """
        获得标注数据
        :param request:
        :return:
        """
        # 返回的字段包括原始文本、预测标签、人工标签
        dataset = request.GET.get("dataset")
        position = request.GET.get("position")
        logging.debug('dataset is %s, position is %s' % (dataset, position))
        # 如果传入有id，则返回id的数据，没有则从0开始
        if position is None:
            position = 0
        classify_data = ClassifyData.objects.values("id", "text", "predict_tag", "human_tag", "status")[int(position)]
        return JsonResponse(classify_data, json_dumps_params={'ensure_ascii': False})

    def post(self, request):
        """
        更改标签/更改状态
        :param request:
        :return:
        """
        dataset = request.GET.get("dataset")
        position = request.GET.get("position")
        tag = request.GET.get("tag")
        status = request.GET.get("status")
        logging.debug('dataset is %s, id is %s, tag is %s, status is %s' % (dataset, position, tag, status))

        data = ClassifyData.objects.filter(id=position)
        if tag is not None:
            data.update(human_tag=tag)
        if status is not None:
            data.update(status=bool(status))
        return JsonResponse({"code": 200}, json_dumps_params={'ensure_ascii': False})


class ClassifyTags(View):
    def get(self, request):
        """
        获得标签
        :param request:
        :return:
        """
        dataset = request.GET.get("dataset")
        tags = ClassifyTag.objects.all()
        if len(tags) == 0:
            return JsonResponse({"code": 200})
        return JsonResponse(tags, json_dumps_params={'ensure_ascii': False})

    def post(self, request):
        """
        增加标签
        :param request:
        :return:
        """
        dataset = request.POST.get("dataset")
        new_tags = json.loads(request.body)
        logging.debug("dataset is %s, tags is %s" % (dataset, new_tags["tags"]))
        tags = ClassifyTag.objects.all()
        tags.delete()
        for tag in new_tags["tags"]:
            tag_model = ClassifyTag(tag=tag)
            tag_model.save()
        return JsonResponse({"code": 200})

    def delete(self, request):
        """
        删除标签
        :return:
        """
        dataset = request.GET.get("dataset")
        tag = request.GET.get("tag")
        logging.debug("dataset is %s, tag is %s" % (dataset, tag))
        tag_set = ClassifyTag.objects.filter(tag=tag)
        tag_set.delete()
        return JsonResponse({"code": 200}, json_dumps_params={'ensure_ascii': False})


@api_view(["GET"])
def classify_stats(request):
    """
    获得统计指标
    :param request:
    :return:
    """
    # 统计准确率、总数据量
    text_num = ClassifyData.objects.count()
    finish_num = ClassifyData.objects.filter(status=True).count()
    correct_num = ClassifyData.objects.filter(human_tag="", status=True).count()
    if finish_num == 0:
        correct_rate = 0
    else:
        correct_rate = correct_num / finish_num * 100
    return JsonResponse({"totalNum": text_num, "finishNum": finish_num, "correctRate": correct_rate}, json_dumps_params={'ensure_ascii': False})
