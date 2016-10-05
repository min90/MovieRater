import csv

# Check for duplicates

print "Search for duplicates in \"cleaned_data.csv\"..."

with open('cleaned_data.csv', 'r') as newfile:
    reader = csv.reader(newfile)
    reader2 = csv.reader(newfile)
    
    for row in reader:
        for row2 in reader2:
            if row[0]==row2[0] and row[1]==row2[1] and row[2]==row2[2] and row[3]==row2[3] and row[4]==row2[4]:
                print "Duplicate found!!"
    
print "Duplicates search finished."
