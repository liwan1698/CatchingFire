# Generated by Django 3.1.5 on 2021-01-08 12:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0003_auto_20210108_1000'),
    ]

    operations = [
        migrations.AlterField(
            model_name='classifydata',
            name='human_tag',
            field=models.CharField(help_text='人工打标签', max_length=30, null=True, verbose_name='人工打标签'),
        ),
        migrations.AlterField(
            model_name='classifydata',
            name='pre_text',
            field=models.TextField(help_text='预处理文本', max_length=5000, null=True, verbose_name='预处理文本'),
        ),
        migrations.AlterField(
            model_name='classifydata',
            name='predict_tag',
            field=models.CharField(help_text='预测标签', max_length=30, null=True, verbose_name='预测标签'),
        ),
        migrations.AlterField(
            model_name='classifydata',
            name='text',
            field=models.TextField(help_text='原始文本', max_length=5000, null=True, verbose_name='原始文本'),
        ),
    ]
