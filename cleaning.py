import csv

# Cleaning process

print "Reading data... (from movie_metadata.csv)"

directors = [] # director = 1
actors1 = [] # actor 1 = 10
actors2 = [] # actor 2 = 6
actors3 = [] # actor 3 = 14
ratings = [] # rating = 25
duplicates = [] # lines to delete

base_rows = 0
duplicate_rows = 0
final_rows = 0

# reading dataset
with open('movie_metadata.csv', 'r') as oldfile:
    reader = csv.reader(oldfile)
    next(reader)

    for row in reader:
        if row[1]=="" or row[10]=="" or row[6]=="" or row[14]=="" or row[25]=="":
            continue
        
        base_rows += 1
        
        # saving in arrays
        directors.append(row[1])
        actors1.append(row[10])
        actors2.append(row[6])
        actors3.append(row[14])
        ratings.append(float(row[25]))

print "\tRows found : " + str(base_rows) + "."
print "Searching for duplicate entries..."

# searching for duplicates
for i in range(0, len(directors)):
    for j in range(0, i):
        if directors[i]==directors[j] and actors1[i]==actors1[j] and actors2[i]==actors2[j] and actors3[i]==actors3[j] and ratings[i]==ratings[j]:
            duplicates.append(i)
            duplicate_rows += 1
            break

print "\tDuplicates found : " + str(duplicate_rows) + "."
print "Saving cleaned data... (into cleaned_data.csv)"

# writing cleaned data into new CSV
with open('cleaned_data.csv', 'w') as newfile:
    writer = csv.writer(newfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Director', 'Actor1', 'Actor2', 'Actor3', 'Rating'])
    
    for k in range(0, len(directors)):
        if k not in duplicates:
            writer.writerow([directors[k], actors1[k], actors2[k], actors3[k], ratings[k]])
            final_rows += 1

print "\tRows saved : " + str(final_rows) + "."
print "Done."
