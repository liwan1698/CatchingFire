# Generated by Django 3.1.5 on 2021-01-08 09:04

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='classifydata',
            name='pending_tag',
        ),
    ]
