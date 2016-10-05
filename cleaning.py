import csv

# Cleaning process

with open('cleaned_data.csv', 'w') as newfile:
    writer = csv.writer(newfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Director', 'Actor1', 'Actor2', 'Actor3', 'Rating'])
    
    with open('movie_metadata.csv', 'r') as oldfile:
        reader = csv.reader(oldfile)
        next(reader)

        # director = 1
        # actor 1 = 10
        # actor 2 = 6
        # actor 3 = 14
        # rating = 25

        for row in reader:
            if row[1]=="" or row[10]=="" or row[6]=="" or row[14]=="" or row[25]=="":
                continue
            
            # saving into the new csv file
            writer.writerow([row[1], row[10], row[6], row[14], float(row[25])])

print "Done."
