## 数据介绍

## Django 准备初始数据库

- cd bookrecommend
- # 在Django项目(my_project)的根目录下执行
- python manage.py startapp app
- python manage.py makemigrations
- python manage.py migrate
- python manage.py createsuperuser # 增加超级管理员
- python manage.py runserver 0.0.0.0:8000 # 启动服务

## 导入数据

- 数据库建表处理
- 1.在MySQL中创建一个database，取好名字，比如MovieData;
- 2.在该数据库中创建moviegenre3和users_resulttable两张表,建表命令行如下：
- CREATE TABLE moviegenre3(imdbId INT NOT NULL PRIMARY KEY,title varchar(300),poster varchar(600));
- CREATE TABLE users_resulttable(userId INT NOT NULL,imdbId INT,rating DECIMAL(3,1));
- load data infile "users_resulttable.csv" into table users_resulttable fields terminated by ',' lines terminated by '
  \n' (userId,imdbId,rating);

