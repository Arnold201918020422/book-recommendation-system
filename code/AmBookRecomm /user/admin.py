from django.contrib import admin

from .models import *


# Register your models here.
class UserAdmin(admin.ModelAdmin):
    list_display = ("username", "password", "name", "address", "email")
    search_fields = ("username", "name", "address", "phone", "email")


class BookAdmin(admin.ModelAdmin):
    list_display = ("title", "author", "brand", 'years', 'description', 'num', 'imageURL')
    search_fields = ("title", "author", "description")
    list_filter = ("brand",)


class ScoreAdmin(admin.ModelAdmin):
    list_display = ("book", "num", "com", "fen")
    search_fields = ("book", "num", "com", "fen")


class ActionAdmin(admin.ModelAdmin):
    def show_all_join(self, obj):
        return [a.name for a in obj.user.all()]

    def num(self, obj):
        return obj.user.count()

    list_display = ("title", "num", "status")
    search_fields = ("title", "content", "user")
    list_filter = ("status",)


class CommenAdmin(admin.ModelAdmin):
    list_display = ("user", "book", "good", "create_time")
    search_fields = ("user", "book", "good")
    list_filter = ("user", "book")


class ActionCommenAdmin(admin.ModelAdmin):
    list_display = ("user", "action", "create_time")
    search_fields = ("user", "action")
    list_filter = ("user", "action")


class LiuyanAdmin(admin.ModelAdmin):
    list_display = ("user", "create_time")
    search_fields = ("user",)
    list_filter = ("user",)


admin.site.register(Category)
admin.site.register(User, UserAdmin)
admin.site.register(Book, BookAdmin)
# admin.site.register(Score, ScoreAdmin)
admin.site.register(Comment, CommenAdmin)
