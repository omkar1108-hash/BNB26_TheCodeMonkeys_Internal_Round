from file_detector import detect_file_type


test_files = [
    "photo.jpg",
    "image.png",
    "notes.txt",
    "PS_Allocation.pdf",
    "report.docx",
    "something.xyz"
]


for filename in test_files:
    result = detect_file_type(filename)
    print(f"{filename} -> {result}")