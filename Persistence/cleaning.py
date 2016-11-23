import csv

# Cleaning process

print ("Reading data... (from movie_metadata.csv)")

directors = [] # director = 1
actors1 = [] # actor 1 = 10
actors2 = [] # actor 2 = 6
actors3 = [] # actor 3 = 14
ratings = [] # rating = 25
facebook_director = []
facebook_actor1 = []
facebook_actor2 = []
facebook_actor3 =[]
duplicates = [] # lines to delete

base_rows = 0
duplicate_rows = 0
final_rows = 0

# reading dataset
with open('../movie_metadata.csv', 'r', encoding="utf8") as oldfile:
    reader = csv.reader(oldfile)
    next(reader)

    for row in reader:
        if row[1]=="" or row[10]=="" or row[6]=="" or row[14]=="" or row[25]=="":
            continue
        
        base_rows += 1
        
        # saving in arrays
        directors.append(row[1])
        facebook_director.append(row[4])
        actors1.append(row[10])
        facebook_actor1.append(row[7])
        actors2.append(row[6])
        facebook_actor2.append(row[24])
        actors3.append(row[14])
        facebook_actor3.append(row[5])
        ratings.append(float(row[25]))

print ("\tRows found : " + str(base_rows) + ".")
print ("Searching for duplicate entries...")

# searching for duplicates
for i in range(0, len(directors)):
    for j in range(0, i):
        if directors[i]==directors[j] and actors1[i]==actors1[j] and actors2[i]==actors2[j] and actors3[i]==actors3[j] and ratings[i]==ratings[j]:
            duplicates.append(i)
            duplicate_rows += 1
            break

print ("\tDuplicates found : " + str(duplicate_rows) + ".")
print ("Saving cleaned data... (into cleaned_data.csv)")

# writing cleaned data into new CSV
with open('cleaned_data.csv', 'w', newline='') as newfile:
    writer = csv.writer(newfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Director', 'Actor1', 'Actor2', 'Actor3', 'Rating', "DirectorLikes", "Actor1Likes", "Actor2Likes", "Actor3Likes"])
    
    for k in range(0, len(directors)):
        if k not in duplicates:
            if facebook_director[k] != "0" and facebook_actor1[k] != "0" and facebook_actor2[k] != "0" and facebook_actor3[k] != "0":
                writer.writerow([directors[k], actors1[k], actors2[k], actors3[k], ratings[k], facebook_director[k], facebook_actor1[k], facebook_actor2[k], facebook_actor3[k]])
                final_rows += 1

print ("\tRows saved : " + str(final_rows) + ".")
print ("Done.")
