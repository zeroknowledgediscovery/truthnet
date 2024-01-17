# Define the path to your file
file_path = '../main.tex'

# Open the file, read its contents, then close the file
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Replace the U+FFFC character with nothing (delete it)
modified_text = text.replace(u"\uFFFC", 'xxxxxxx')

# Write the modified text back to the file
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(modified_text)

print("The Unicode character U+FFFC has been removed from the file.")
