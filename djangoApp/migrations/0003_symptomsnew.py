# Generated by Django 4.0.3 on 2022-04-11 00:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('djangoApp', '0002_rename_name_symptoms_name'),
    ]

    operations = [
        migrations.CreateModel(
            name='SymptomsNew',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('NewName', models.CharField(max_length=25)),
            ],
        ),
    ]