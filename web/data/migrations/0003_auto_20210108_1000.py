# Generated by Django 3.1.5 on 2021-01-08 10:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0002_remove_classifydata_pending_tag'),
    ]

    operations = [
        migrations.CreateModel(
            name='ClassifyTag',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('tag', models.CharField(default='', help_text='标签种类', max_length=30, verbose_name='标签种类')),
            ],
        ),
        migrations.AddField(
            model_name='classifydata',
            name='status',
            field=models.BooleanField(default=False, help_text='是否完成', verbose_name='是否完成'),
        ),
    ]
