from django.urls import path
from . import views
from .views import GUI_start


#from . import main
from . import trial
from . import images_views

from django.conf.urls.static import static
from django.conf import settings

app_name = 'BirdProgram'

urlpatterns = [
    path('', views.GUI_start, name='GUI_start'),
#    path('program1/', main.step_1, name='main_program1'),
    path('program1/', trial.program1, name='main_program1'),
    path('program2/', trial.program2, name='main_program2'),
    path('program3/', images_views.image_view, name='main_program3'),
    path('program4/', images_views.bird_ok, name='bird_ok')
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)