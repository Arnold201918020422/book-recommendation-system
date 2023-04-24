from django.urls import path

from user import views

urlpatterns = [
    path("", views.index, name="index"),
    path("most_view/", views.most_view, name="most_view"),
    path("all_books/", views.all_books, name="all_books"),
    path("most_mark/", views.most_mark, name="most_mark"),
    # # 收藏最多
    path("hot_book/", views.hot_book, name="hot_book"),
    path("login/", views.login, name="login"),
    path("regist/", views.regist, name="regist"),
    path("logout/", views.logout, name="logout"),
    path("person/", views.personal, name="person"),
    path("kindof/", views.kindof, name="kindof"),
    path("kind/<int:kind_id>/", views.kind, name="kind"),
    path("book/<int:book_id>/", views.book, name="book"),
    path("my_collect/", views.my_collect, name="my_collect"),
    path("my_comments/", views.my_comments, name="my_comments"),
    path("my_rate/", views.my_rate, name="my_rate"),
    # path("item/", views.item, name="item"),
    path("score/<int:book_id>/", views.score, name="score"),
    path("good/<int:commen_id>/<int:book_id>/", views.good, name="good"),
    path("collect/<int:book_id>/", views.collect, name="collect"),
    path("decollect/<int:book_id>/", views.decollect, name="decollect"),
    path("comment/<int:book_id>/", views.commen, name="comment"),
    # path("myjoin/", views.myjoin, name="myjoin"),
    path("delete_comment/<int:comment_id>", views.delete_comment, name="delete_comment"),
    path("delete_rate/<int:rate_id>", views.delete_rate, name="delete_rate"),
    path("recommend/", views.recommend, name="recommend"),
    # path("latest_music/", views.latest_music, name="latest_music"),
    # path("search/", views.search, name="search"),
    # path("golden_horse/", views.golden_horse, name="golden_horse"),
    # path("oscar/", views.oscar, name="oscar"),
    # path("begin/", views.begin, name="begin"),
    # path("week_reco/", views.reco_by_week, name="week_reco"),
    # path("monthitem/", views.reco_by_week, name="monthitem"),
]
