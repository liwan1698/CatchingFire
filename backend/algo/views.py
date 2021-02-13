from django.db.models import Q, F
from rest_framework.decorators import api_view
from django.views.generic.base import View
from django.http import JsonResponse

from .model.bert_bilstm_crf import BertBilstmCrf
from .model.process_data import DataProcess
from .models import ClassifyData, ClassifyTag, NerData
import logging
import json

from .model.fasttext import FastText


class Classify(View):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.human_tag_num = 0   # 记录人工修改数
        self.real_time_train_num = 100   # 当HUMAN_TAG_NUM达到100，则触发实时训练

    def get(self, request):
        """
        获得标注数据
        :param request:
        :return:
        """
        # 返回的字段包括原始文本、预测标签、人工标签
        begin_num = int(request.GET.get("begin_num"))
        fitch_num = int(request.GET.get("fitch_num"))
        logging.info('begin_num is %d, fitch_num is %d' % (begin_num, fitch_num))
        if fitch_num > 100:
            fitch_num = 100
        classify_data = ClassifyData.objects.values("id", "text", "predict_tag", "human_tag")[
                        begin_num:begin_num+fitch_num]
        return JsonResponse(list(classify_data), json_dumps_params={'ensure_ascii': False}, safe=False)

    def post(self, request):
        """
        更改标签
        :param request:
        :return:
        """
        id = int(request.GET.get("id"))
        tag = int(request.GET.get("tag"))
        logging.info('id is %d, tag is %d' % (id, tag))

        data = ClassifyData.objects.filter(id=id)
        data.update(human_tag=tag)
        self.human_tag_num += 1
        # 触发实时训练
        if self.human_tag_num == self.real_time_train_num:
            fasttext = FastText(realtime_train=True, data_index=[id-self.real_time_train_num, id])
            fasttext.build()
            fasttext.train()
            fasttext.predict_all()
            self.human_tag_num = 0
        return JsonResponse({"code": 200}, json_dumps_params={'ensure_ascii': False})


class ClassifyTags(View):
    def get(self, request):
        """
        获得标签
        :param request:
        :return:
        """
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
        new_tags = json.loads(request.body)
        logging.debug("tags is %s" % (new_tags["tags"]))
        tags = ClassifyTag.objects.all()
        tags.delete()
        for tag in new_tags["tags"]:
            tag_model = ClassifyTag(tag=tag)
            tag_model.save()
        return JsonResponse({"code": 200})

    # def delete(self, request):
    #     """
    #     删除标签
    #     :return:
    #     """
    #     tag = request.GET.get("tag")
    #     logging.debug("tag is %s" % (tag))
    #     tag_set = ClassifyTag.objects.filter(tag=tag)
    #     tag_set.delete()
    #     return JsonResponse({"code": 200}, json_dumps_params={'ensure_ascii': False})


@api_view(["GET"])
def classify_stats(request):
    """
    获得统计指标
    :param request:
    :return:
    """
    # 统计准确率、总数据量
    text_num = ClassifyData.objects.count()
    finish_data = ClassifyData.objects.filter(~Q(human_tag=None))
    finish_num = finish_data.count()
    correct_num = finish_data.filter(human_tag=F('predict_tag')).count()
    if finish_num == 0:
        correct_rate = 0
    else:
        correct_rate = correct_num / finish_num * 100
    return JsonResponse({"totalNum": text_num, "finishNum": finish_num, "correctRate": correct_rate}, json_dumps_params={'ensure_ascii': False})


@api_view(["POST"])
def clear_classify_data(request):
    """
    清除分类的数据、模型、标签
    :param request:
    :return:
    """
    pass


class Ner(View):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.human_tag_num = 0   # 记录人工修改数
        self.real_time_train_num = 100   # 当HUMAN_TAG_NUM达到100，则触发实时训练

    def get(self, request):
        """
        获得标注数据
        :param request:
        :return:
        """
        # 返回的字段包括原始文本、预测标签、人工标签
        begin_num = int(request.GET.get("begin_num"))
        fitch_num = int(request.GET.get("fitch_num"))
        logging.info('begin_num is %d, fitch_num is %d' % (begin_num, fitch_num))
        if fitch_num > 100:
            fitch_num = 100
        classify_data = NerData.objects.values("id", "text", "predict_label", "human_label")[
                        begin_num:begin_num+fitch_num]
        return JsonResponse(list(classify_data), json_dumps_params={'ensure_ascii': False}, safe=False)

    def post(self, request):
        """
        更改标签
        :param request: body = {"id": 1, "poses": [{"begin": 2, "end": 3, "pos": "LOC"}]}
        :return:
        """
        logging.info('body is %s' % (request.body))
        tag = json.loads(request.body)
        # 整理标注数据存储
        id = tag['id']
        poses = tag['poses']
        data = NerData.objects.filter(id=id)
        data.update(human_label=json.dumps(poses))
        self.human_tag_num += 1
        # 触发实时训练
        if self.human_tag_num == self.real_time_train_num:
            # todo
            max_len = 100
            dp = DataProcess(max_len=max_len)
            # todo 改为从数据库读取数据
            train_data, train_label, test_data, test_label = dp.get_data(one_hot=True)
            model = BertBilstmCrf(dp.vocab_size, dp.tag_size, max_len=max_len)
            model.build()
            model.train()
            model.predict_all()
            self.human_tag_num = 0
        return JsonResponse({"code": 200}, json_dumps_params={'ensure_ascii': False})
