from django.db import models


class Diseases_Symptoms(models.Model):
    Disease = models.CharField(max_length=50)
    Symptom = models.CharField(max_length=50, unique=True)
    # pub_date = models.DateTimeField('date published')
    def __str__(self):
		    return (str(self.Symptom))


# class Choice(models.Model):
#     question = models.ForeignKey(Question, on_delete=models.CASCADE)
#     choice_text = models.CharField(max_length=200)
#     votes = models.IntegerField(default=0)