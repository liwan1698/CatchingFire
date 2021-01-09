from django.db import models


class ClassifyData(models.Model):
    """
    文本分类标注数据，包括原始数据、预处理后数据、预测结果、人工标注结果、待定标记
    """
    text = models.TextField(null=True, max_length=5000, verbose_name="原始文本", help_text="原始文本")
    pre_text = models.TextField(null=True, max_length=5000, verbose_name="预处理文本", help_text="预处理文本")
    predict_tag = models.CharField(null=True, max_length=30, verbose_name="预测标签", help_text="预测标签")
    human_tag = models.CharField(null=True, max_length=30, verbose_name="人工打标签", help_text="人工打标签")
    status = models.BooleanField(default=False, verbose_name="是否完成", help_text="是否完成")


class ClassifyTag(models.Model):
    """
    文本分类的标签
    """
    tag = models.CharField(default="", max_length=30, verbose_name="标签种类", help_text="标签种类")
