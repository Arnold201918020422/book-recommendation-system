
import pandas as pd
import numpy as np


#Users #read the csv file properly

u_cols = ['user_id', 'location', 'age']
Users_data = pd.read_csv("/Users/chengfujia/Desktop/project/code/implement/user.csv", sep=';', names=u_cols, encoding='latin-1',low_memory=False, na_values=" NaN", on_bad_lines='skip')

#Books

i_cols = ['isbn', 'book_title' ,'book_author','year_of_publication', 'publisher', 'img_s', 'img_m', 'img_l']
Books_data = pd.read_csv("/Users/chengfujia/Desktop/project/code/implement/books.csv",  sep=';', names=i_cols, encoding='latin-1',low_memory=False,na_values=" NaN", on_bad_lines='skip')

#Ratings

r_cols = ['user_id', 'isbn', 'rating']
Ratings_data = pd.read_csv("/Users/chengfujia/Desktop/project/code/implement/ranking.csv",  sep=';', names=r_cols, encoding='latin-1',low_memory=False,  on_bad_lines='skip')

# delete the first Useless data
Users_data = Users_data.drop(Users_data.index[0])
Books_data = Books_data.drop(Books_data.index[0])
Ratings_data =Ratings_data.drop(Ratings_data.index[0])

# Ratings_data  = pd.get_dummies(Ratings_data)

# delete weird data

df1=Books_data[Books_data['year_of_publication'].str.contains(pat='DK Publishing Inc',regex=False)]
df2=Books_data[Books_data['year_of_publication'].str.contains(pat='DK Publishing Inc',regex=False)]  

df2=Books_data.drop(Books_data[Books_data['year_of_publication'].str.contains(pat='DK Publishing Inc',regex=False)].index,inplace=True)

df1=Books_data.drop(Books_data[Books_data['year_of_publication'].str.contains(pat='Gallimard',regex=False)].index,inplace=True)


# convert the type of data
Users_data['age'] = Users_data['age'].astype(float)
Users_data['user_id'] = Users_data['user_id'].astype(int)
Ratings_data['user_id'] = Ratings_data['user_id'].astype(int)
Ratings_data['rating'] = Ratings_data['rating'].astype(float)
Books_data['year_of_publication'] = Books_data['year_of_publication'].astype(float)



# delete no-sense ages
Users_data.loc[(Users_data.age>99) | (Users_data.age<5),'age'] = np.nan
Users_data.age = Users_data.age.fillna(Users_data.age.mean())
# fix the missing data
#publisher
Books_data.loc[Books_data.isbn=='193169656X','publisher']='Mundania Press LLC'
Books_data.loc[Books_data.isbn=='1931696993','publisher']='Novelbooks Incorporated'

#book_author

Books_data.loc[Books_data.isbn=='9627982032','book_author']='Larissa Anne Downe'

#check
# print(Users_data.isnull().sum())
# print(Users_data['age'].describe())
# print(Users_data['user_id'].describe())
# print(Ratings_data['user_id'].describe())
# print(Ratings_data['rating'].describe())
# print(Books_data['year_of_publication'].describe())
# print(Users_data)
# print(Books_data)
# print(Ratings_data)


#check the null value
print(Ratings_data.isnull().sum())
# print(Books_data.isnull().sum())
# print(Books_data.loc[Books_data.publisher.isnull(),:])
# print(Books_data.loc[Books_data.book_author.isnull(),:])


# delete the wrong time
Books_data.loc[(Books_data.year_of_publication==0)|(Books_data.year_of_publication>2021) ,'year_of_publication' ] = np.nan
Books_data.year_of_publication = Books_data.year_of_publication.fillna(round(Books_data.year_of_publication.mean()))
#check the wrong time data
# print(sorted(Books_data['year_of_publication'].unique()))

# mix data
md = pd.merge(Users_data, Ratings_data, on='user_id')
md = pd.merge(md, Books_data, on='isbn')
# print(md.head(5))

#store
# Users_data.head()
Users_data.to_csv("user.csv", index=False)
Books_data.to_csv("books.csv", index=False)
Ratings_data.to_csv("ranking.csv", index=False)
md.to_csv('mix_data.csv', index=False)



