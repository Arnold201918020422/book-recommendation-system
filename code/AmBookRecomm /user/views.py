import json
import os
import pickle
from functools import wraps
from operator import itemgetter

from django.core.cache import cache
from django.core.paginator import Paginator
from django.db.models import Count, Avg
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt

from cache_keys import USER_CACHE, ITEM_CACHE
from user.forms import Edit
from user.models import Book, User, Category, Rate, Comment


def pageList(curPage, totalPage, displayPage):
    if totalPage > 0:
        span = int(displayPage / 2)
        if curPage < span:
            return list(range(1, totalPage + 1)[:displayPage])
        else:
            return list(range(curPage - span + 1, totalPage + 1)[:displayPage])
    else:
        return []


def book_paginator(books, page, pageSize=10):
    paginator = Paginator(books, pageSize)
    if page is None:
        page = 1
    return paginator.page(page)


@csrf_exempt
def regist(request):
    resJson = {'msg': 'OK', 'code': 203}
    if request.method == "POST":
        paramJson = json.loads(request.body.decode())
        username = paramJson['username']
        email = paramJson['email']
        password = paramJson['password']
        print("regist ===>", username, "===>", password)
        result = User.objects.filter(username=username)
        if result:
            resJson['code'] = 201
            resJson['error'] = "用户已存在"
        else:
            newUser = User()
            newUser.username = username
            newUser.password = password
            newUser.email = email
            newUser.name = username
            newUser.save()
            resJson['code'] = 200
        return JsonResponse(resJson)
    else:
        return render(request, "user/regit.html")


def logout(request):
    if not request.session.get("login_in", None):  # 不在登录状态跳转回首页
        return redirect(reverse("index"))
    request.session.flush()  # 清除session信息
    return redirect(reverse("index"))


@csrf_exempt
def login(request):
    resJson = {'msg': 'OK', 'code': 200}
    if request.method == "POST":
        paramJson = json.loads(request.body.decode())
        username = paramJson['username']
        password = paramJson['password']
        print("login ===>", username, "===>", password)
        result = User.objects.filter(username=username)
        if result:
            user = User.objects.get(username=username)
            if user.password == password:
                request.session["login_in"] = True
                request.session["user_id"] = user.id
                request.session["username"] = user.username
                request.session["name"] = user.name
                resJson['code'] = 200
                return JsonResponse(resJson)
        resJson['code'] = 201
        return JsonResponse(resJson)
    else:
        return render(request, "user/login.html")


def login_in(func):  # 验证用户是否登录
    @wraps(func)
    def wrapper(*args, **kwargs):
        request = args[0]
        is_login = request.session.get("login_in")
        if is_login:
            return func(*args, **kwargs)
        else:
            return redirect(reverse("login"))

    return wrapper


@login_in
def my_collect(request):
    user = User.objects.get(id=request.session.get("user_id"))
    books = user.book_set.all()
    return render(request, "user/my_collect.html", {"books": books})


@login_in
def my_comments(request):
    user = User.objects.get(id=request.session.get("user_id"))
    comments = user.comment_set.all()
    print('comment:', comments)
    return render(request, "user/my_comment.html", {"comments": comments})


@login_in
def my_rate(request):
    user = User.objects.get(id=request.session.get("user_id"))
    rate = user.rate_set.all()
    return render(request, "user/my_rate.html", {"rate": rate})


@login_in
def personal(request):
    user = User.objects.get(id=request.session.get("user_id"))
    if request.method == "POST":
        form = Edit(instance=user, data=request.POST)
        if form.is_valid():
            form.save()
            return redirect(reverse("personal"))
        else:
            return render(
                request, "user/personal.html", {"message": "修改失败", "form": form}
            )
    form = Edit(instance=user)
    return render(request, "user/personal.html", {"form": form})


# Create your views here.
def index(request):
    return render(request, "index.html")


def all_books(request):
    books = Book.objects.annotate(user_collector=Count('collect')).order_by('-user_collector')
    current_page = request.GET.get("page", 1)
    books = book_paginator(books, current_page, 10)
    book1 = books[0] if books else None
    pages = pageList(books.number, books.paginator.num_pages, displayPage=10)
    print("ALL:", books, "TOTAL:", books.paginator.num_pages)
    return render(request, "user/items.html", {"books": books, "book1": book1, "pages": pages, })


def most_view(request):
    current_page = request.GET.get("page", 1)
    books = Book.objects.annotate(user_collector=Count('num')).order_by('-num')
    books = book_paginator(books, current_page, 10)
    book1 = books[0] if books else None
    pages = pageList(books.number, books.paginator.num_pages, displayPage=10)
    print("MOST:", pages, "TOTAL:", books.paginator.num_pages)
    return render(request, "user/most.html", {"mostBooks": books, "mostBook1": book1, "mostPages": pages})


def most_mark(request):
    page_number = request.GET.get("page", 1)
    books = Book.objects.all().annotate(num_mark=Count('rate')).order_by('-num_mark')
    books = book_paginator(books, page_number, pageSize=10)
    book1 = books[0] if books else None
    pages = pageList(books.number, books.paginator.num_pages, displayPage=10)
    print("RATE:", pages, "TOTAL:", books.paginator.num_pages)
    return render(request, "user/rate.html", {"rateBooks": books, "rateBook1": book1, "ratePages": pages})


def hot_book(request):
    # page_number = request.GET.get("page", 1)

    page_number = request.GET.get("page", 1)
    books = Book.objects.annotate(user_collector=Count('collect')).order_by('-user_collector')
    books = book_paginator(books, page_number, pageSize=10)
    book1 = books[0] if books else None
    pages = pageList(books.number, books.paginator.num_pages, displayPage=10)
    print("HOT:", pages, "TOTAL:", books.paginator.num_pages)
    return render(request, "user/hot.html", {"hotBooks": books, "hotBook1": book1, "hotPages": pages})


def kindof(request):
    tags = Category.objects.all().annotate(num_book=Count('book')).order_by('-num_book')
    tag = tags.first()
    # tags = Tags.objects.get(id=tag.id)
    books = tag.book_set.all()
    print("kindof ==>", len(books))
    page_number = request.GET.get("page", 1)
    books = book_paginator(books, page_number, pageSize=12)
    book1 = books[0] if books else None
    pages = pageList(books.number, books.paginator.num_pages, displayPage=12)
    print("HOT:", pages, "TOTAL:", books.paginator.num_pages)
    return render(request, "user/kindof.html",
                  {"tags": tags[1:7], "kindBooks": books, "pages": pages, "kindBook1": book1})


def kind(request, kind_id):
    tags = Category.objects.all().annotate(num_book=Count('book')).order_by('-num_book')
    tag = Category.objects.get(id=kind_id)
    books = tag.book_set.all()
    print("kindof ==>", len(books))
    page_number = request.GET.get("page", 1)
    books = book_paginator(books, page_number, pageSize=12)
    book1 = books[0] if books else None
    pages = pageList(books.number, books.paginator.num_pages, displayPage=12)
    return render(request, "user/kindof.html",
                  {"tags": tags[1:7], "kindBooks": books, "pages": pages, "kindBook1": book1})

@login_in
def book(request, book_id):
    book = Book.objects.get(pk=book_id)
    book.num += 1
    book.save()
    comments = book.comment_set.order_by("-create_time")
    user_id = request.session.get("user_id")

    book_rate = Rate.objects.filter(book=book).all().aggregate(Avg('mark'))
    if book_rate:
        book_rate = book_rate['mark__avg']
    if book_id is not None:
        user_rate = Rate.objects.filter(book=book, user_id=user_id).first()
        user = User.objects.get(pk=user_id)
        is_collect = book.collect.filter(id=user_id).first()
    return render(request, "user/book.html", locals())


@login_in
# 在打分的时候清楚缓存
def score(request, book_id):
    user_id = request.session.get("user_id")
    # user = User.objects.get(id=user_id)
    book = Book.objects.get(id=book_id)
    score = float(request.POST.get("score"))
    get, created = Rate.objects.get_or_create(user_id=user_id, book=book, defaults={"mark": score})
    if created:
        print('create data')
        # 清理缓存
        user_cache = USER_CACHE.format(user_id=user_id)
        item_cache = ITEM_CACHE.format(user_id=user_id)
        cache.delete(user_cache)
        cache.delete(item_cache)
        print('cache deleted')

    return redirect(reverse("book", args=(book_id,)))


@login_in
def collect(request, book_id):
    user = User.objects.get(id=request.session.get("user_id"))
    book = Book.objects.get(id=book_id)
    book.collect.add(user)
    book.save()
    return redirect(reverse("book", args=(book_id,)))


@login_in
def decollect(request, book_id):
    user = User.objects.get(id=request.session.get("user_id"))
    book = Book.objects.get(id=book_id)
    book.collect.remove(user)
    book.save()
    return redirect(reverse("book", args=(book_id,)))


@login_in
def commen(request, book_id):
    user = User.objects.get(id=request.session.get("user_id"))
    book = Book.objects.get(id=book_id)
    comment = request.POST.get("comment")
    Comment.objects.create(user=user, book=book, content=comment)
    return redirect(reverse("book", args=(book_id,)))


def good(request, commen_id, book_id):
    commen = Comment.objects.get(id=commen_id)
    commen.good += 1
    commen.save()
    return redirect(reverse("book", args=(book_id,)))


@login_in
def delete_comment(request, comment_id):
    Comment.objects.get(pk=comment_id).delete()
    return redirect(reverse("my_comments"))


def delete_rate(request, rate_id):
    Rate.objects.filter(pk=rate_id).delete()
    return redirect(reverse("my_rate"))


def dcn_rec(user_id, K=30):
    print("dcn_rec-->", user_id, os.getcwd())
    with open('data/ucf.pkl', mode='rb') as f:
        userSimMatrix = pickle.load(f)
    recommends = dict()
    # 先获取user具有正反馈的item数组
    hist_rates = Rate.objects.filter(user_id=user_id)
    related_items = [r.book.id for r in hist_rates]
    print("related_items:", related_items)
    # 将其他用户与user按照相似度逆序排序之后取前K个
    for v, sim in sorted(userSimMatrix[user_id].items(), key=itemgetter(1), reverse=True)[:K]:
        # 从与user相似的用户的喜爱列表中寻找可能的物品进行推荐
        v_hist_rates = Rate.objects.filter(user_id=v)
        print("V:", v, v_hist_rates)
        for item in [r.book.id for r in v_hist_rates]:
            # 如果与user相似的用户喜爱的物品与user喜欢的物品重复了，直接跳过
            if item in related_items:
                continue
            recommends.setdefault(item, 0.)
            recommends[item] += sim
    # 根据被推荐物品的相似度逆序排列，然后推荐前N个物品给到用户
    ret = dict(sorted(recommends.items(), key=itemgetter(1), reverse=True))
    return [int(mid) for mid in ret]


@login_in
def recommend(request):
    page = request.GET.get("page", 1)
    user_id = request.session.get("user_id")
    cache_key = USER_CACHE.format(user_id=user_id)
    book_list = cache.get(cache_key)
    if not book_list:
        book_ids = dcn_rec(user_id)
        # music_list = Music.objects.filter(pk__in=music_ids)
        records = []
        for mid in book_ids:
            tmp_music = Book.objects.get(id=mid)
            records.append(tmp_music)
            # print(mid, tmp_music)
        if not records:
            records = Book.objects.all().order_by('?')[:100]
        cache.set(cache_key, records, 60 * 5)
        # print('reco_by_week => music_ids:', music_ids)
        print('设置缓存->', cache_key, len(book_ids))
    else:
        print('user {}缓存命中!'.format(user_id))
    book_list = cache.get(cache_key)
    books = book_paginator(book_list, page)
    book1 = books[0] if books else None
    pages = pageList(books.number, books.paginator.num_pages, displayPage=10)
    print("推荐书籍", books)
    path = request.path
    title = "推荐书籍"
    return render(
        request, "user/items.html",
        {"books": books, "path": path, "title": title, "pages": pages, "book1": book1}
    )
