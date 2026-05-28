from django.urls import path
from app import views

urlpatterns = [
    path('',views.home,name='home'),
    path('home/',views.home,name='home'),
    path('about/',views.about,name='about'),
    path('detection/',views.detection,name='detection'),
    path('dataset/',views.dataset,name='dataset'),
    path('dataset/classify/', views.classify_unknowns, name='classify_unknowns'),
    path('dataset/assign/', views.assign_classified_group, name='assign_classified_group'),
    path('video_feed/',views.video_feed,name='video_feed'),
    path('video_stats/',views.video_stats,name='video_stats'),
    path('login/',views.login_user,name='login'),
    path('login/face/feed/', views.face_login_feed, name='face_login_feed'),
    path('login/face/check/', views.face_login_check, name='face_login_check'),
    path('logout/',views.logout_user,name='logout'),
    path('forgot_password/',views.forgot_password,name='forgot_password'),
    path('delete_person/<int:person_id>/', views.delete_person, name='delete_person'),
    path('delete_image/<int:image_id>/', views.delete_image, name='delete_image'),
    path('reports/', views.reports_view, name='reports'),
    path('reports/export/', views.export_reports_excel, name='export_reports_excel'),
    path('test_alert/', views.test_alert, name='test_alert'),
]


