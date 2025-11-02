import os

def print_structure(root, indent=""):
    for item in os.listdir(root):
        path = os.path.join(root, item)
        print(indent + "- " + item)
        if os.path.isdir(path):
            print_structure(path, indent + "  ")

print_structure(".")

# This script prints the directory structure of the current working directory.
