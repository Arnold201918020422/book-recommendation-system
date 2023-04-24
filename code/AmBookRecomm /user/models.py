from django.db import models
from django.db.models import Avg


class User(models.Model):
    username = models.CharField(max_length=128, unique=True, verbose_name="账号")
    password = models.CharField(max_length=128, verbose_name="密码")
    phone = models.CharField(max_length=128, verbose_name="手机号码")
    name = models.CharField(max_length=128, verbose_name="名字")
    address = models.CharField(max_length=128, verbose_name="地址")
    email = models.EmailField(verbose_name="邮箱")

    class Meta:
        verbose_name_plural = "用户列表"
        verbose_name = "用户列表"

    def __str__(self):
        return self.name


class Category(models.Model):
    name = models.CharField(max_length=128, verbose_name="标签", unique=True)

    class Meta:
        verbose_name = "书籍标签"
        verbose_name_plural = "书籍标签"

    def __str__(self):
        return self.name


class Book(models.Model):
    tags = models.ManyToManyField(Category, verbose_name='标签', blank=True)
    collect = models.ManyToManyField(User, verbose_name="收藏者", blank=True)
    sump = models.IntegerField(verbose_name="收藏人数", default=0)
    asin = models.CharField(verbose_name="书籍编号", max_length=512, unique=True)
    title = models.CharField(verbose_name="书籍名称", max_length=512, unique=False)
    description = models.CharField(verbose_name='书籍描述', max_length=512)
    author = models.CharField(verbose_name="作者", max_length=512)
    brand = models.CharField(verbose_name="品牌", max_length=512)
    years = models.CharField(verbose_name="年份", max_length=512)
    num = models.IntegerField(verbose_name="浏览量", default=0)
    imageURL = models.URLField(verbose_name="封面图片", max_length=512)

    class Meta:
        verbose_name = "书籍列表"
        verbose_name_plural = "书籍列表"

    def __str__(self):
        return self.asin + "[" + self.title + "#@" + self.author + "#@" + self.brand + "#@" + self.years + "]"


class Rate(models.Model):
    book = models.ForeignKey(
        Book, on_delete=models.CASCADE, blank=True, null=True, verbose_name="书籍id"
    )
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, blank=True, null=True, verbose_name="用户id",
    )
    mark = models.FloatField(verbose_name="评分")
    create_time = models.DateTimeField(verbose_name="发布时间", auto_now_add=True)

    @property
    def avg_mark(self):
        average = Rate.objects.all().aggregate(Avg('mark'))['mark__avg']
        return average

    class Meta:
        verbose_name = "评分信息"
        verbose_name_plural = verbose_name


class Comment(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="用户")
    content = models.CharField(max_length=64, verbose_name="内容")
    create_time = models.DateTimeField(auto_now_add=True)
    good = models.IntegerField(verbose_name="点赞", default=0)
    book = models.ForeignKey(Book, on_delete=models.CASCADE, verbose_name="书籍")

    class Meta:
        verbose_name = "评论"
        verbose_name_plural = verbose_name
