import csv

# Cleaning process

print("Reading data... (from movie_metadata.csv)")

all_directors = []
all_actors = []

directors = [] # director = 1
actors1 = [] # actor 1 = 10
actors2 = [] # actor 2 = 6
actors3 = [] # actor 3 = 14
ratings = [] # rating = 25
duplicates = [] # lines to delete

total_rows = 0
uncomplete_rows = 0
base_rows = 0
duplicate_rows = 0
final_rows = 0

# reading dataset
with open('movie_metadata.csv', 'r') as oldfile:
    reader = csv.reader(oldfile)
    next(reader)

    for row in reader:
        # rows in original dataset
        total_rows += 1

        if row[1]=="" or row[10]=="" or row[6]=="" or row[14]=="" or row[25]=="":
            # uncomplete rows (missing actor / rating /...)
            uncomplete_rows += 1
            continue

        # saved rows
        base_rows += 1

        if row[1] not in all_directors:
            all_directors.append(row[1])
        if row[10] not in all_actors:
            all_actors.append(row[10])
        if row[6] not in all_actors:
            all_actors.append(row[6])
        if row[14] not in all_actors:
            all_actors.append(row[14])
        
        # saving in arrays
        directors.append(row[1])
        actors1.append(row[10])
        actors2.append(row[6])
        actors3.append(row[14])
        ratings.append(float(row[25]))

print("\tTotal number of directors : " + str(len(all_directors)) + ".")
print("\tTotal number of actors : " + str(len(all_actors)) + ".")
print("\tRows found : " + str(total_rows) + ".")
print("\tUncomplete rows (removed) : " + str(uncomplete_rows) + ".")
print("\tRows saved : " + str(base_rows) + ".")
print("Searching for duplicate entries...")

# searching for duplicates
for i in range(0, len(directors)):
    for j in range(0, i):
        if directors[i]==directors[j] and actors1[i]==actors1[j] and actors2[i]==actors2[j] and actors3[i]==actors3[j] and ratings[i]==ratings[j]:
            duplicates.append(i)
            duplicate_rows += 1
            break

print("\tDuplicates found : " + str(duplicate_rows) + ".")
print("Saving cleaned data... (into cleaned_data.csv)")

# writing cleaned data into new CSV
with open('cleaned_data.csv', 'w') as newfile:
    writer = csv.writer(newfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Director', 'Actor1', 'Actor2', 'Actor3', 'Director_ID', 'Actor1_ID', 'Actor2_ID', 'Actor3_ID', 'Rating'])
    
    for k in range(0, len(directors)):
        if k not in duplicates:
            dir_ID = all_directors.index(directors[k])
            actor1_ID = all_actors.index(actors1[k])
            actor2_ID = all_actors.index(actors2[k])
            actor3_ID = all_actors.index(actors3[k])

            writer.writerow([directors[k], actors1[k], actors2[k], actors3[k], dir_ID, actor1_ID, actor2_ID, actor3_ID, ratings[k]])
            final_rows += 1

print("\tRows saved : " + str(final_rows) + ".")
print("Done.")
