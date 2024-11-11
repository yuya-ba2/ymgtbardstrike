from django import forms
from .models import Article
from .models import Upload


#class UploadForm(forms.ModelForm):
#    class Meta:
#        model = Upload
#        fields = ['file']

class UploadForm(forms.Form):
    file = forms.FileField()
#    files = forms.FileField(widget=forms.FileInput(attrs={'multiple': True}))

#class SearchForm(forms.Form):
#    keyword = forms.CharField(label='検索', max_length=100)

class ArticleForm(forms.ModelForm):
    class Meta:
        model = Article
        fields = ('content', 'user_name')
        