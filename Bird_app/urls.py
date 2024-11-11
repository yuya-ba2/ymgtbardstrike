
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('BirdProgram/', include('BirdProgram.urls')),
    path('admin/', admin.site.urls),
]
