import csv

# Check for duplicates

print ("Search for duplicates in \"cleaned_data.csv\"...")

directors = []
actors1 = []
actors2 = []
actors3 = []
ratings = []

with open('cleaned_data.csv', 'r') as newfile:
    reader = csv.reader(newfile)
    
    loop = 0
    nb = 0
    
    for row in reader:
        for i in range(0, len(ratings)):
            loop += 1
            if row[0]==directors[i] and row[1]==actors1[i] and row[2]==actors2[i] and row[3]==actors3[i] and row[4]==ratings[i]:
                nb += 1
        
        directors.append(row[0])
        actors1.append(row[1])
        actors2.append(row[2])
        actors3.append(row[3])
        ratings.append(row[4])
    
    print(str(loop) + " loop turns.")
    
    if nb == 0:
        print("There is no duplicate entries.")
    else:
        print(str(nb) + " duplicate entries.")
    
print("End.")
