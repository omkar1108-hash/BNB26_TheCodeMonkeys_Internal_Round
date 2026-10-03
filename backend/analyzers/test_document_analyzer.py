from pathlib import Path

from document_analyzer import extract_document_text


uploads_dir = Path("../../data/uploads")

files = list(uploads_dir.iterdir())

if not files:
    print("No uploaded files found.")
    exit()


for file_path in files:

    if file_path.suffix.lower() in [".pdf", ".docx", ".txt"]:

        print("=" * 60)
        print(f"FILE: {file_path.name}")
        print("=" * 60)

        try:
            text = extract_document_text(str(file_path))

            print(text[:5000])

            print("\n")
            print(f"Characters extracted: {len(text)}")

        except Exception as e:
            print(f"ERROR: {e}")