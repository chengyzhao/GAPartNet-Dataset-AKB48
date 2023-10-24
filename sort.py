def read_and_sort(filename):
    # Create empty lists for each category
    buckets, boxes, drawers, trashcans = [], [], [], []
    
    # Read the file and classify lines
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  # Remove trailing newline
            if line.startswith("Bucket"):
                buckets.append(line)
            elif line.startswith("Box"):
                boxes.append(line)
            elif line.startswith("Drawer"):
                drawers.append(line)
            elif line.startswith("TrashCan"):
                trashcans.append(line)

    # Sort each list
    buckets.sort(key=lambda x: int(x.split(" ")[1]))
    boxes.sort(key=lambda x: int(x.split(" ")[1]))
    drawers.sort(key=lambda x: int(x.split(" ")[1]))
    trashcans.sort(key=lambda x: int(x.split(" ")[1]))

    # Return the sorted lists combined in original format
    return boxes + buckets + drawers + trashcans

# Use the function
filename = "./configs/final_id_akb48_new.txt"
sorted_list = read_and_sort(filename)

# Output the sorted result
with open(filename, 'w') as file:
    for item in sorted_list:
        file.write(item + '\n')

print("File has been sorted!")
